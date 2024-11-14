import enum
import json
import logging
import re
from typing import Annotated

from kani import AIFunction, AIParam, ChatMessage, FunctionCall, Kani, ToolCall, ai_function, chat_in_terminal
from kani.engines import Completion, WrapperEngine
from kani.engines.huggingface import HuggingEngine

log = logging.getLogger("qwen-fc")


class Unit(enum.Enum):
    FAHRENHEIT = "fahrenheit"
    CELSIUS = "celsius"


class MyKani(Kani):
    @ai_function()
    def get_weather(
        self,
        location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
        unit: Unit,
    ):
        """Get the current weather in a given location."""
        # call some weather API, or just mock it for this example
        degrees = 72 if unit == Unit.FAHRENHEIT else 22
        return f"Weather in {location}: Sunny, {degrees} degrees {unit.value}."


class QwenFunctionCallingAdapter(WrapperEngine):
    def __init__(self, *args, tool_call_start="<tool_call>", tool_call_end="</tool_call>", eos="<|im_end|>", **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_call_start = tool_call_start
        self.tool_call_end = tool_call_end
        self.eos = eos

    def _parse_tool_calls(self, content: str) -> tuple[str, list[ToolCall]]:
        last_end = 0
        tool_calls = []
        content_parts = []
        matches = re.finditer(
            rf"{re.escape(self.tool_call_start)}\s*(.+?)\s*{re.escape(self.tool_call_end)}",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        for tool_json in matches:
            log.debug(f"Found tool JSON while parsing: {tool_json.group(1)}")
            action = json.loads(tool_json.group(1))
            content_parts.append(content[last_end : tool_json.pos])
            last_end = tool_json.endpos + 1

            # translate back to kani spec
            tool_name = action["name"]
            tool_args = json.dumps(action["arguments"])
            tool_call = ToolCall.from_function_call(FunctionCall(name=tool_name, arguments=tool_args))
            tool_calls.append(tool_call)

        # return trimmed content and tool calls
        return "\n".join(content_parts).strip(), tool_calls

    async def predict(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams):
        hyperparams.setdefault("decode_kwargs", dict(skip_special_tokens=False))
        completion = await super().predict(messages, functions, **hyperparams)

        # if we have tools, parse
        if functions:
            completion.message.content, completion.message.tool_calls = self._parse_tool_calls(completion.message.text)
        completion.message.content = completion.message.content.removesuffix(self.eos).strip()

        return completion

    async def stream(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams):
        content_parts = []
        in_tool_call = False
        has_seen_tool_call = False
        inner_completion = None
        hyperparams.setdefault("decode_kwargs", dict(skip_special_tokens=False))

        # consume from the inner iterator, yielding as normal until we see a tool call or a completion
        async for elem in super().stream(messages, functions, **hyperparams):
            log.debug(f"Got stream element: {elem!r}")
            if isinstance(elem, str):
                content_parts.append(elem)
                # if we see the start of a tool call, stop yielding and start buffering
                if self.tool_call_start in elem:
                    yield elem[: elem.index(self.tool_call_start)]
                    in_tool_call = True
                    has_seen_tool_call = True
                # otherwise yield the string
                if not in_tool_call:
                    yield elem.removesuffix(self.eos)
                # if we see the end of a tool call, start yielding again
                if self.tool_call_end in elem:
                    in_tool_call = False
                    yield elem[elem.index(self.tool_call_end) + len(self.tool_call_end) :].removesuffix(self.eos)
            else:
                # save the inner completion
                inner_completion = elem

        # we have consumed all the elements - construct a new completion
        # if we don't have a tool call we can just yield the inner completion
        if not has_seen_tool_call and inner_completion:
            yield inner_completion
        # otherwise, parse tool calls from the content (preserving inner tool calls if necessary)
        else:
            content = "".join(content_parts)
            log.debug(f"Content before parsing tool calls: {content!r}")
            content, tool_calls = self._parse_tool_calls(content)
            if inner_completion:
                tool_calls = (inner_completion.message.tool_calls or []) + tool_calls
                prompt_tokens = inner_completion.prompt_tokens
                completion_tokens = inner_completion.completion_tokens
            else:
                prompt_tokens = None
                completion_tokens = None
            clean_content = content.removesuffix(self.eos).strip()
            yield Completion(
                ChatMessage.assistant(clean_content, tool_calls=tool_calls),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


model = HuggingEngine(model_id="Qwen/Qwen2.5-72B-Instruct")
engine = QwenFunctionCallingAdapter(model)
ai = MyKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai, verbose=True, stream=True)
