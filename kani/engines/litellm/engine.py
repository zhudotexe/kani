import logging
from collections.abc import AsyncIterable

from kani.ai_function import AIFunction
from kani.engines.base import BaseCompletion, BaseEngine, Completion
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall

try:
    import litellm
except ImportError:
    raise MissingModelDependencies(
        'The LiteLLMEngine requires extra dependencies. Please install kani with "pip install kani[litellm]".'
    ) from None

log = logging.getLogger(__name__)


def _kani_msg_to_dict(msg: ChatMessage) -> dict:
    """Translate a kani ChatMessage to an OpenAI-format dict for litellm."""
    if msg.role == ChatRole.FUNCTION and msg.tool_call_id is not None:
        return {"role": "tool", "content": msg.text, "tool_call_id": msg.tool_call_id}
    if msg.role == ChatRole.FUNCTION:
        return {"role": "function", "name": msg.name or "", "content": msg.text}

    d = {"role": msg.role.value, "content": msg.text}
    if msg.name:
        d["name"] = msg.name
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def _translate_functions(functions: list[AIFunction]) -> list[dict]:
    """Translate kani AIFunctions to OpenAI-format tool specs."""
    return [
        {"type": "function", "function": {"name": f.name, "description": f.desc, "parameters": f.json_schema}}
        for f in functions
    ]


def _parse_tool_calls(raw_tool_calls) -> list[ToolCall]:
    """Parse litellm tool call objects into kani ToolCalls."""
    result = []
    for tc in raw_tool_calls:
        func = tc.function if hasattr(tc, "function") else tc.get("function", {})
        name = func.name if hasattr(func, "name") else func.get("name", "")
        args = func.arguments if hasattr(func, "arguments") else func.get("arguments", "")
        tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
        tc_type = tc.type if hasattr(tc, "type") else tc.get("type", "function")
        result.append(ToolCall(id=tc_id, type=tc_type, function=FunctionCall(name=name, arguments=args)))
    return result


class LiteLLMEngine(BaseEngine):
    """Engine for using any LLM provider via the LiteLLM SDK.

    LiteLLM supports 100+ providers (OpenAI, Anthropic, Azure, Bedrock, Vertex AI, Groq, etc.)
    through a unified ``litellm.acompletion()`` interface.

    Models use the ``provider/model-name`` format (e.g. ``anthropic/claude-haiku-4-5``).
    Provider API keys are read from environment variables (e.g. ``ANTHROPIC_API_KEY``).
    """

    def __init__(
        self,
        model: str,
        max_context_size: int = 4096,
        *,
        api_key: str = None,
        api_base: str = None,
        retry: int = 3,
        **hyperparams,
    ):
        """
        :param model: The model to use in litellm format (e.g. ``anthropic/claude-haiku-4-5``,
            ``openai/gpt-4o``, ``bedrock/anthropic.claude-3-haiku``).
        :param max_context_size: The maximum number of tokens allowed in the chat prompt.
        :param api_key: The API key for the provider. If not set, litellm reads from environment
            variables (e.g. ``ANTHROPIC_API_KEY``).
        :param api_base: Custom API base URL (e.g. for Azure or self-hosted endpoints).
        :param retry: How many times to retry failed requests (default 3).
        :param hyperparams: Additional parameters passed to every ``litellm.acompletion()`` call.
        """
        self.model = model
        self.max_context_size = max_context_size
        self.api_key = api_key
        self.api_base = api_base
        self.retry = retry
        self.hyperparams = hyperparams

        litellm.num_retries = retry

    def prompt_len(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **kwargs) -> int:
        translated = [_kani_msg_to_dict(m) for m in messages]
        try:
            count = litellm.token_counter(model=self.model, messages=translated)
        except Exception:
            count = sum(len((m.get("content") or "").split()) * 2 for m in translated)
        if functions:
            import json

            tool_str = json.dumps(_translate_functions(functions))
            try:
                count += litellm.token_counter(model=self.model, text=tool_str)
            except Exception:
                count += len(tool_str.split()) * 2
        return count

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> BaseCompletion:
        translated = [_kani_msg_to_dict(m) for m in messages]
        kwargs = {
            "model": self.model,
            "messages": translated,
            "drop_params": True,
            **self.hyperparams,
            **hyperparams,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if functions:
            kwargs["tools"] = _translate_functions(functions)

        response = await litellm.acompletion(**kwargs)

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""

        tool_calls = None
        if msg.tool_calls:
            tool_calls = _parse_tool_calls(msg.tool_calls)

        kani_msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        prompt_tokens = getattr(response.usage, "prompt_tokens", None) if response.usage else None
        completion_tokens = getattr(response.usage, "completion_tokens", None) if response.usage else None

        return Completion(
            message=kani_msg,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def stream(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        translated = [_kani_msg_to_dict(m) for m in messages]
        kwargs = {
            "model": self.model,
            "messages": translated,
            "stream": True,
            "drop_params": True,
            **self.hyperparams,
            **hyperparams,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if functions:
            kwargs["tools"] = _translate_functions(functions)

        response = await litellm.acompletion(**kwargs)

        content_chunks = []
        tool_call_partials = {}

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content_chunks.append(delta.content)
                yield delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_partials:
                        tool_call_partials[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_call_partials[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_partials[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_partials[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = None
        if tool_call_partials:
            tool_calls = [
                ToolCall(id=p["id"], type="function", function=FunctionCall(name=p["name"], arguments=p["arguments"]))
                for p in tool_call_partials.values()
            ]

        final_content = "".join(content_chunks) or None
        kani_msg = ChatMessage(role=ChatRole.ASSISTANT, content=final_content, tool_calls=tool_calls)

        yield Completion(message=kani_msg)
