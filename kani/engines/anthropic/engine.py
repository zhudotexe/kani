import functools
import json
import os

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole
from kani.prompts.pipeline import PromptPipeline
from ..base import BaseEngine, Completion

try:
    from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic

    # anthropic async client loads a small json file using anyio for some reason; hook into the underlying loader
    # noinspection PyProtectedMember
    from anthropic._tokenizers import sync_get_tokenizer
except ImportError as e:
    raise MissingModelDependencies(
        'The AnthropicEngine requires extra dependencies. Please install kani with "pip install kani[anthropic]".'
    ) from None

CONTEXT_SIZES_BY_PREFIX = [
    ("claude-3", 200000),
    ("claude-2.1", 200000),
    ("", 100000),
]


# ==== pipe ====
def content_transform(msg: ChatMessage):
    # FUNCTION messages should look like:
    # {
    #   "role": "user",
    #   "content": [
    #     {
    #       "type": "tool_result",
    #       "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
    #       "content": "65 degrees"
    #     }
    #   ]
    # }
    if msg.role != ChatRole.FUNCTION:
        return msg.text
    # todo is_error
    result = {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.text}
    return [result]


# assumes system messages are plucked before calling
CLAUDE_PIPELINE = (
    PromptPipeline()
    .translate_role(role=ChatRole.SYSTEM, to=ChatRole.USER)
    .merge_consecutive(role=ChatRole.USER, sep="\n")
    .merge_consecutive(role=ChatRole.ASSISTANT, sep=" ")
    .ensure_bound_function_calls()
    .ensure_start(role=ChatRole.USER)
    .conversation_dict(function_role="user", content_transform=content_transform)
)


class AnthropicEngine(BaseEngine):
    """Engine for using the Anthropic API.

    This engine supports all Claude models. See https://docs.anthropic.com/claude/docs/getting-access-to-claude for
    information on accessing the Claude API.

    See https://docs.anthropic.com/claude/reference/selecting-a-model for a list of available models.
    """

    # because we have to estimate tokens wildly and the ctx is so long we'll just reserve a bunch
    token_reserve = 500

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-3-haiku",
        max_tokens: int = 512,
        max_context_size: int = None,
        *,
        retry: int = 2,
        api_base: str = None,
        headers: dict = None,
        client: AsyncAnthropic = None,
        **hyperparams,
    ):
        """
        :param api_key: Your Anthropic API key. By default, the API key will be read from the `ANTHROPIC_API_KEY`
            environment variable.
        :param model: The id of the model to use (e.g. "claude-2.1", "claude-instant-1.2").
        :param max_tokens: The maximum number of tokens to sample at each generation (defaults to 512).
            Generally, you should set this to the same number as your Kani's ``desired_response_tokens``.
        :param max_context_size: The maximum amount of tokens allowed in the chat prompt. If None, uses the given
            model's full context size.
        :param retry: How many times the engine should retry failed HTTP calls with exponential backoff (default 2).
        :param api_base: The base URL of the Anthropic API to use.
        :param headers: A dict of HTTP headers to include with each request.
        :param client: An instance of ``anthropic.AsyncAnthropic`` (for reusing the same client in multiple engines).
            You must specify exactly one of (api_key, client). If this is passed the ``retry``, ``api_base``,
            and ``headers`` params will be ignored.
        :param hyperparams: Any additional parameters to pass to the underlying API call (see
            https://docs.anthropic.com/claude/reference/complete_post).
        """
        if api_key and client:
            raise ValueError("You must supply no more than one of (api_key, client).")
        if api_key is None and client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "You must supply an `api_key`, `client`, or set the `OPENAI_API_KEY` environment variable to use"
                    " the OpenAIEngine."
                )
        if max_context_size is None:
            max_context_size = next(size for prefix, size in CONTEXT_SIZES_BY_PREFIX if model.startswith(prefix))
        self.client = client or AsyncAnthropic(
            api_key=api_key, max_retries=retry, base_url=api_base, default_headers=headers
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams

        # token counting - claude 3+ does not release tokenizer so we have to do heuristics and cache
        self.token_cache = {}
        if model.startswith("claude-2"):
            self.tokenizer = sync_get_tokenizer()
        else:
            # claude 3 tokenizer just... doesn't exist
            # https://github.com/anthropics/anthropic-sdk-python/issues/375 pain
            self.tokenizer = None

    # ==== token counting ====
    @staticmethod
    def message_cache_key(message: ChatMessage):
        # (role, content, tool calls)

        # we'll use msgpart identity for the hash here since we'll always have a ref as long as it's in a message
        # history
        hashable_content = tuple(part if isinstance(part, str) else id(part) for part in message.parts)

        # use (name, args) for tool calls
        if message.tool_calls:
            hashable_tool_calls = tuple((tc.function.name, tc.function.arguments) for tc in message.tool_calls)
        else:
            hashable_tool_calls = message.tool_calls

        return hash((message.role, hashable_content, hashable_tool_calls))

    def message_len(self, message: ChatMessage) -> int:
        # use cache
        cache_key = self.message_cache_key(message)
        if cache_key in self.token_cache:
            return self.token_cache[cache_key]

        # use tokenizer
        if self.tokenizer is not None:
            return self._message_len_tokenizer(message)

        # panik - I guess we'll pretend that 4 chars = 1 token...?
        n = len(message.role.value) + len(message.text)
        if message.tool_calls:
            for tc in message.tool_calls:
                n += len(tc.function.name) + len(tc.function.arguments)
        return n // 4

    def _message_len_tokenizer(self, message):
        # human messages are prefixed with `\n\nHuman: ` and assistant with `\n\nAssistant:`
        if message.role == ChatRole.USER:
            mlen = 5
        elif message.role == ChatRole.ASSISTANT:
            mlen = 4
        else:
            mlen = 2  # we'll prepend system/function messages with \n\n as a best-effort case

        if message.text:
            mlen += len(self.tokenizer.encode(message.text).ids)
        return mlen

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0
        # wrap an inner impl to use lru_cache with frozensets
        return self._function_token_reserve_impl(frozenset(functions))

    @functools.lru_cache(maxsize=256)
    def _function_token_reserve_impl(self, functions):
        # panik, also assume len/4?
        n = sum(len(f.name) + len(f.desc) + len(json.dumps(f.json_schema)) for f in functions)
        return n // 4

    # ==== requests ====
    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        kwargs = {}

        # --- messages ---
        # pluck system messages
        last_system_idx = next((i for i, m in enumerate(messages) if m.role != ChatRole.SYSTEM), None)
        if last_system_idx:
            kwargs["system"] = "\n\n".join(m.text for m in messages[:last_system_idx])
            messages = messages[last_system_idx:]

        # enforce ordering and function call bindings
        # and translate to dict spec
        messages = CLAUDE_PIPELINE(messages)

        # --- tools ---
        if functions:
            kwargs["tools"] = [
                {"name": f.name, "description": f.desc, "input_schema": f.json_schema} for f in functions
            ]

        completion = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            **kwargs,
            **self.hyperparams,
            **hyperparams,
        )

        # todo translate to kani
        return Completion(message=ChatMessage.assistant(completion.completion.strip()))

    async def close(self):
        await self.client.close()
