import functools

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage
from . import function_calling
from .translation import ChatCompletion, translate_functions, translate_messages
from ..base import BaseEngine

try:
    import tiktoken
    from openai import AsyncOpenAI as OpenAIClient
except ImportError as e:
    raise MissingModelDependencies(
        'The OpenAIEngine requires extra dependencies. Please install kani with "pip install kani[openai]".'
    ) from None

# https://platform.openai.com/docs/models
CONTEXT_SIZES_BY_PREFIX = [
    ("gpt-3.5-turbo-0125", 16384),
    ("gpt-3.5-turbo-1106", 16384),
    ("gpt-3.5-turbo-16k", 16384),
    ("gpt-3.5-turbo", 4096),
    # gpt-4-turbo models aren't prefixed differently...
    # TODO make the default gpt-4 128k and keep the pre-turbo ones at 8k after gpt-4 defaults to 128k
    ("gpt-4-1106", 128000),
    ("gpt-4-0125", 128000),
    ("gpt-4-vision", 128000),
    ("gpt-4-turbo", 128000),
    ("gpt-4-32k", 32768),
    ("gpt-4", 8192),
    # fine-tunes
    ("ft:gpt-3.5-turbo-16k", 16384),
    ("ft:gpt-3.5-turbo", 4096),
    ("ft:gpt-4-32k", 32768),
    ("ft:gpt-4", 8192),
    # completion models
    ("text-davinci-", 4096),
    ("code-", 8000),
    # catch-all
    ("", 2048),  # e.g. aba/babbage/curie/davinci
]


class OpenAIEngine(BaseEngine):
    """Engine for using the OpenAI API.

    This engine supports all chat-based models and fine-tunes.
    """

    def __init__(
        self,
        api_key: str = None,
        model="gpt-3.5-turbo",
        max_context_size: int = None,
        *,
        organization: str = None,
        retry: int = 5,
        api_base: str = "https://api.openai.com/v1",
        headers: dict = None,
        client: OpenAIClient = None,
        **hyperparams,
    ):
        """
        :param api_key: Your OpenAI API key. By default, the API key will be read from the `OPENAI_API_KEY` environment
            variable.
        :param model: The id of the model to use (e.g. "gpt-3.5-turbo", "ft:gpt-3.5-turbo:my-org:custom_suffix:id").
        :param max_context_size: The maximum amount of tokens allowed in the chat prompt. If None, uses the given
            model's full context size.
        :param organization: The OpenAI organization to use in requests. By default, the org ID would be read from the
            `OPENAI_ORG_ID` environment variable (defaults to the API key's default org if not set).
        :param retry: How many times the engine should retry failed HTTP calls with exponential backoff (default 5).
        :param api_base: The base URL of the OpenAI API to use.
        :param headers: A dict of HTTP headers to include with each request.
        :param client: An instance of `openai.AsyncOpenAI <https://github.com/openai/openai-python>`_
            (for reusing the same client in multiple engines).
            You must specify exactly one of ``(api_key, client)``. If this is passed the ``organization``, ``retry``,
            ``api_base``, and ``headers`` params will be ignored.
        :param hyperparams: The arguments to pass to the ``create_chat_completion`` call with each request. See
            https://platform.openai.com/docs/api-reference/chat/create for a full list of params.
        """
        if api_key and client:
            raise ValueError("You must supply no more than one of (api_key, client).")
        if max_context_size is None:
            max_context_size = next(size for prefix, size in CONTEXT_SIZES_BY_PREFIX if model.startswith(prefix))
        self.client = client or OpenAIClient(
            api_key=api_key, organization=organization, max_retries=retry, base_url=api_base, default_headers=headers
        )
        self.model = model
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams
        self.tokenizer = None  # tiktoken caches a tokenizer globally in module, so we can unconditionally load it
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def message_len(self, message: ChatMessage) -> int:
        mlen = 7
        if message.text:
            mlen += len(self.tokenizer.encode(message.text))
        if message.name:
            mlen += len(self.tokenizer.encode(message.name))
        if message.tool_calls:
            for tc in message.tool_calls:
                mlen += len(self.tokenizer.encode(tc.function.name))
                mlen += len(self.tokenizer.encode(tc.function.arguments))
        return mlen

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> ChatCompletion:
        if functions:
            tool_specs = translate_functions(functions)
        else:
            tool_specs = None
        # translate to openai spec - group any tool messages together and ensure all free ToolCall IDs are bound
        translated_messages = translate_messages(messages)
        # make API call
        completion = await self.client.chat.completions.create(
            model=self.model, messages=translated_messages, tools=tool_specs, **self.hyperparams, **hyperparams
        )
        # translate into Kani spec and return
        return ChatCompletion(openai_completion=completion)

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0
        # wrap an inner impl to use lru_cache with frozensets
        return self._function_token_reserve_impl(frozenset(functions))

    @functools.lru_cache(maxsize=256)
    def _function_token_reserve_impl(self, functions):
        # openai doesn't tell us exactly how their function prompt works, so
        # we rely on community reverse-engineering to build the right prompt
        # hopefully OpenAI releases a utility to calculate this in the future, this seems kind of fragile
        prompt = function_calling.prompt(functions)
        return len(self.tokenizer.encode(prompt)) + 16  # internal MD headers, namespace {} delimiters

    async def close(self):
        await self.client.close()
