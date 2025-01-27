import json

from kani.engines import Completion
from kani.models import ChatMessage, FunctionCall, ToolCall
from .base import BaseToolCallParser


class NaiveJSONToolCallParser(BaseToolCallParser):
    """
    If the model's output contains only valid JSON of form:

    .. code-block:: json

        {
            "name": "function_name",
            "parameters": {
                "key": "value..."
            }
        }

    then assume it is a function call. Otherwise, return the content unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, tool_call_start_token="", tool_call_end_token="", **kwargs)

    def parse_tool_calls(self, content: str) -> tuple[str, list[ToolCall]]:
        """Given the string completion of the model, return the content without tool calls and the parsed tool calls."""
        try:
            data = json.loads(content.strip())
            match data:
                case {"name": str(name), "parameters": dict(parameters)}:
                    tc = ToolCall.from_function_call(FunctionCall.with_args(name, **parameters))
                    return "", [tc]
        except json.JSONDecodeError:
            return content, []
        return content, []

    async def stream(self, messages, functions=None, **hyperparams):
        # special case - if we see a { at start of message, defer until end of message to see if it's a function call
        # otherwise stream as normal
        content_parts = []
        seen_non_tool_call_token = False
        in_tool_call = False
        inner_completion = None

        # consume from the inner iterator, yielding as normal until we see a tool call or a completion
        async for elem in self.engine.stream(messages, functions, **hyperparams):
            if isinstance(elem, str):
                content_parts.append(elem)
                # if we see {, stop yielding and start buffering
                if elem.lstrip().startswith("{") and not seen_non_tool_call_token:
                    in_tool_call = True
                # otherwise yield the string
                if elem and not in_tool_call:
                    seen_non_tool_call_token = True
                    yield elem
            else:
                # save the inner completion
                inner_completion = elem

        # we have consumed all the elements - construct a new completion
        # if we don't have a tool call we can just yield the inner completion
        if not in_tool_call and inner_completion:
            yield inner_completion
        # otherwise, parse tool calls from the content (preserving inner tool calls if necessary)
        else:
            content = "".join(content_parts)
            # noinspection DuplicatedCode
            content, tool_calls = self.parse_tool_calls(content)
            if inner_completion:
                tool_calls = (inner_completion.message.tool_calls or []) + tool_calls
                prompt_tokens = inner_completion.prompt_tokens
                completion_tokens = inner_completion.completion_tokens
            else:
                prompt_tokens = None
                completion_tokens = None
            clean_content = content.strip()
            yield Completion(
                ChatMessage.assistant(clean_content, tool_calls=tool_calls),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
