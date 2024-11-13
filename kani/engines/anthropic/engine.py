import functools
import itertools
import json
import logging
import os
from typing import AsyncIterable

from kani.ai_function import AIFunction
from kani.exceptions import KaniException, MissingModelDependencies
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall
from kani.prompts.pipeline import PromptPipeline
from ..base import BaseCompletion, BaseEngine, Completion
from ..mixins import TokenCached

try:
    from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic
except ImportError as e:
    raise MissingModelDependencies(
        'The AnthropicEngine requires extra dependencies. Please install kani with "pip install kani[anthropic]".'
    ) from None

CONTEXT_SIZES_BY_PREFIX = [
    ("claude-3", 200000),
    ("claude-2.1", 200000),
    ("", 100000),
]

log = logging.getLogger(__name__)


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
    if msg.role == ChatRole.FUNCTION:
        result = {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.text}

        # tool call error
        if msg.is_tool_call_error:
            result["is_error"] = True

        return [result]

    # ASSISTANT messages with tool calls should look like:
    # {
    #   "role": "assistant",
    #   "content": [
    #     {
    #       "type": "text",
    #       "text": "<thinking>I need to use the get_weather, and the user wants San Francisco, CA.</thinking>"
    #     },
    #     {
    #       "type": "tool_use",
    #       "id": "toolu_01A09q90qw90lq917835lq9",
    #       "name": "get_weather",
    #       "input": {"location": "San Francisco, CA", "unit": "celsius"}
    #     }
    #   ]
    # }
    if msg.role == ChatRole.ASSISTANT and msg.tool_calls:
        content = [{"type": "text", "text": str(part)} for part in msg.parts]
        for tc in msg.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": tc.function.kwargs})
        return content

    return msg.text


# assumes system messages are plucked before calling
CLAUDE_PIPELINE = (
    PromptPipeline()
    .translate_role(role=ChatRole.SYSTEM, to=ChatRole.USER)
    .merge_consecutive(role=ChatRole.USER, sep="\n")
    .merge_consecutive(role=ChatRole.ASSISTANT, sep=" ")
    .ensure_bound_function_calls()
    .conversation_dict(function_role="user", content_transform=content_transform)
)


class AnthropicEngine(TokenCached, BaseEngine):
    """Engine for using the Anthropic API.

    This engine supports all Claude models. See https://docs.anthropic.com/claude/docs/getting-access-to-claude for
    information on accessing the Claude API.

    See https://docs.anthropic.com/claude/docs/models-overview for a list of available models.
    """

    # because we have to estimate tokens wildly and the ctx is so long we'll just reserve a bunch
    token_reserve = 500

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-3-haiku-20240307",
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
                    "You must supply an `api_key`, `client`, or set the `ANTHROPIC_API_KEY` environment variable to use"
                    " the OpenAIEngine."
                )
        if max_context_size is None:
            max_context_size = next(size for prefix, size in CONTEXT_SIZES_BY_PREFIX if model.startswith(prefix))

        super().__init__()

        self.client = client or AsyncAnthropic(
            api_key=api_key, max_retries=retry, base_url=api_base, default_headers=headers
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams

        # token counting - claude 3+ does not release tokenizer so we have to do heuristics and cache
        if model.startswith("claude-2"):
            # anthropic async client loads a json file using anyio for some reason; hook into the underlying loader
            # noinspection PyProtectedMember
            from anthropic._tokenizers import sync_get_tokenizer

            self.tokenizer = sync_get_tokenizer()
        else:
            # claude 3 tokenizer just... doesn't exist
            # https://github.com/anthropics/anthropic-sdk-python/issues/375 pain
            self.tokenizer = None

    # ==== token counting ====
    def message_len(self, message: ChatMessage) -> int:
        if (cached_len := self.get_cached_message_len(message)) is not None:
            return cached_len

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
        # this only applies to claude-2
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
    @staticmethod
    def _prepare_request(messages, functions):
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

        # merge FUNCTION, USER consecutives into one with multiple parts
        prompt_msgs = []
        for role, group_msgs in itertools.groupby(messages, key=lambda m: m["role"]):
            group_msgs = list(group_msgs)
            # >1 consecutive user messages get merged
            if role == "user" and len(group_msgs) > 1:
                # turn str parts into {type: text, text: ...}
                prompt_msg_content = []
                for msg in group_msgs:
                    if isinstance(msg["content"], str):
                        prompt_msg_content.append({"type": "text", "text": msg["content"]})
                    else:
                        prompt_msg_content.extend(msg["content"])
                # and output the final msg
                prompt_msgs.append({"role": "user", "content": prompt_msg_content})
            # else send to output
            else:
                prompt_msgs.extend(group_msgs)

        # --- tools ---
        if functions:
            kwargs["tools"] = [
                {"name": f.name, "description": f.desc, "input_schema": f.json_schema} for f in functions
            ]

        log.debug(f"Claude message format: {prompt_msgs}")

        return kwargs, prompt_msgs

    def _translate_anthropic_message(self, message):
        tool_calls = []
        text_parts = []
        for part in message.content:
            if part.type == "text":
                text_parts.append(part.text)
            elif part.type == "tool_use":
                fc = FunctionCall(name=part.name, arguments=json.dumps(part.input))
                tc = ToolCall(id=part.id, type="function", function=fc)
                tool_calls.append(tc)
            else:
                raise KaniException(
                    f"The engine returned an unknown part: {part.type}. Please report this as an issue on GitHub with"
                    " instructions on how to reproduce this issue."
                )
        content = text_parts[0] if len(text_parts) == 1 else text_parts
        kani_msg = ChatMessage.assistant(content, tool_calls=tool_calls or None)

        # also cache the message token len
        self.set_cached_message_len(kani_msg, message.usage.output_tokens)

        return Completion(
            message=kani_msg,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        kwargs, prompt_msgs = self._prepare_request(messages, functions)

        # --- completion ---
        assert len(prompt_msgs) > 0
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=prompt_msgs,
            **kwargs,
            **self.hyperparams,
            **hyperparams,
        )

        # translate to kani
        return self._translate_anthropic_message(message)

    async def stream(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        # do the stream
        kwargs, prompt_msgs = self._prepare_request(messages, functions)

        assert len(prompt_msgs) > 0
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=prompt_msgs,
            **kwargs,
            **self.hyperparams,
            **hyperparams,
        ) as stream:
            async for text in stream.text_stream:
                yield text

            message = await stream.get_final_message()
            yield self._translate_anthropic_message(message)

    async def close(self):
        await self.client.close()
