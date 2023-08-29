import functools
import os

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage
from . import function_calling
from .client import OpenAIClient
from .models import ChatCompletion, FunctionSpec
from ..base import BaseEngine

try:
    import tiktoken
except ImportError as e:
    raise MissingModelDependencies(
        'The OpenAIEngine requires extra dependencies. Please install kani with "pip install kani[openai]".'
    ) from None

# https://platform.openai.com/docs/models
CONTEXT_SIZES_BY_PREFIX = [
    ("gpt-3.5-turbo-16k", 16384),
    ("gpt-3.5-turbo", 4096),
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
        :param organization: The OpenAI organization to use in requests (defaults to the API key's default org).
        :param retry: How many times the engine should retry failed HTTP calls with exponential backoff (default 5).
        :param api_base: The base URL of the OpenAI API to use.
        :param headers: A dict of HTTP headers to include with each request.
        :param client: An instance of :class:`.OpenAIClient` (for reusing the same client in multiple engines). You must
            specify exactly one of (api_key, client). If this is passed the ``organization``, ``retry``, ``api_base``,
            and ``headers`` params will be ignored.
        :param hyperparams: Any additional parameters to pass to
            :meth:`.OpenAIClient.create_chat_completion`.
        """
        if api_key and client:
            raise ValueError("You must supply no more than one of (api_key, client).")
        if api_key is None and client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "You must supply an `api_key`, `client`, or set the `OPENAI_API_KEY` environment variable to use"
                    " the OpenAIEngine."
                )
        if max_context_size is None:
            max_context_size = next(size for prefix, size in CONTEXT_SIZES_BY_PREFIX if model.startswith(prefix))
        self.client = client or OpenAIClient(
            api_key, organization=organization, retry=retry, api_base=api_base, headers=headers
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
        mlen = 5  # ChatML = 4, role = 1
        if message.content:
            mlen += len(self.tokenizer.encode(message.content))
        if message.name:
            mlen += len(self.tokenizer.encode(message.name))
        if message.function_call:
            mlen += len(self.tokenizer.encode(message.function_call.name))
            mlen += len(self.tokenizer.encode(message.function_call.arguments))
        return mlen

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> ChatCompletion:
        if functions:
            function_spec = [FunctionSpec(name=f.name, description=f.desc, parameters=f.json_schema) for f in functions]
        else:
            function_spec = None
        completion = await self.client.create_chat_completion(
            model=self.model, messages=messages, functions=function_spec, **self.hyperparams, **hyperparams
        )
        return completion

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
