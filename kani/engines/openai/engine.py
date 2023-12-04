import functools
import os

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies, PromptError
from kani.models import ChatMessage, ChatRole
from . import function_calling
from .client import OpenAIClient
from .models import ChatCompletion, FunctionSpec, OpenAIChatMessage, ToolSpec
from ..base import BaseEngine

try:
    import tiktoken
except ImportError as e:
    raise MissingModelDependencies(
        'The OpenAIEngine requires extra dependencies. Please install kani with "pip install kani[openai]".'
    ) from None

# https://platform.openai.com/docs/models
CONTEXT_SIZES_BY_PREFIX = [
    ("gpt-3.5-turbo-1106", 16384),
    ("gpt-3.5-turbo-16k", 16384),
    ("gpt-3.5-turbo", 4096),
    ("gpt-4-1106", 128000),
    ("gpt-4-vision", 128000),
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
        mlen = 7
        if message.text:
            mlen += len(self.tokenizer.encode(message.text))
        if message.name:
            mlen += len(self.tokenizer.encode(message.name))
        if message.function_call:
            mlen += len(self.tokenizer.encode(message.function_call.name))
            mlen += len(self.tokenizer.encode(message.function_call.arguments))
        return mlen

    # translation helpers
    @staticmethod
    def translate_functions(functions: list[AIFunction], cls: type[ToolSpec] = ToolSpec) -> list[ToolSpec]:
        return [
            cls.from_function(FunctionSpec(name=f.name, description=f.desc, parameters=f.json_schema))
            for f in functions
        ]

    @staticmethod
    def translate_messages(
        messages: list[ChatMessage], cls: type[OpenAIChatMessage] = OpenAIChatMessage
    ) -> list[OpenAIChatMessage]:
        translated_messages = []
        free_toolcall_ids = set()
        for m in messages:
            # if this is not a function result and there are free tool call IDs, raise
            if m.role != ChatRole.FUNCTION and free_toolcall_ids:
                raise PromptError(
                    f"Encountered a {m.role.value!r} message but expected a FUNCTION message to satisfy the pending"
                    f" tool call(s): {free_toolcall_ids}"
                )
            # asst: add tool call IDs to freevars
            if m.role == ChatRole.ASSISTANT and m.tool_calls:
                for tc in m.tool_calls:
                    free_toolcall_ids.add(tc.id)
            # func: bind freevars
            elif m.role == ChatRole.FUNCTION:
                # has ID: bind it if requested; translate to FUNCTION if not
                if m.tool_call_id is not None:
                    if m.tool_call_id in free_toolcall_ids:
                        free_toolcall_ids.remove(m.tool_call_id)
                    else:
                        # this happens if the tool call is pushed out of context but the result is still here,
                        # and we have always included messages beforehand
                        # TODO: this will eventually be deprecated - maube we just skip this message?
                        m = m.copy_with(tool_call_id=None)
                # no ID: bind if unambiguous, otherwise cry
                elif m.tool_call_id is None:
                    if len(free_toolcall_ids) == 1:
                        m = m.copy_with(tool_call_id=free_toolcall_ids.pop())
                    elif len(free_toolcall_ids) > 1:
                        raise PromptError(
                            "Got a FUNCTION message with no tool_call_id but multiple tool calls are pending"
                            f" ({free_toolcall_ids})! Set the tool_call_id to resolve the pending tool requests."
                        )
            translated_messages.append(cls.from_chatmessage(m))
        # if the translated messages start with a hanging TOOL call, strip it (openai limitation)
        # though hanging FUNCTION messages are OK
        while translated_messages and translated_messages[0].role == "tool":
            translated_messages.pop(0)
        return translated_messages

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> ChatCompletion:
        if functions:
            tool_specs = self.translate_functions(functions)
        else:
            tool_specs = None
        # translate to openai spec - group any tool messages together and ensure all free ToolCall IDs are bound
        translated_messages = self.translate_messages(messages)
        # make API call
        completion = await self.client.create_chat_completion(
            model=self.model, messages=translated_messages, tools=tool_specs, **self.hyperparams, **hyperparams
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
