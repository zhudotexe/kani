import json

from kani.models import FunctionCall, ToolCall
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
        super().__init__(*args, tool_call_start_token=None, tool_call_end_token=None, **kwargs)

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
        seen_non_tool_call_token = False
        in_tool_call = False

        # consume from the inner iterator, yielding as normal until we see a tool call or a completion
        async for elem in super().stream(messages, functions, **hyperparams):
            if isinstance(elem, str):
                # if we see {, stop yielding and start buffering
                if elem.lstrip().startswith("{") and not seen_non_tool_call_token:
                    in_tool_call = True
                # otherwise yield the string
                if elem and not in_tool_call:
                    seen_non_tool_call_token = True
                    yield elem
            else:
                # yield the inner completion
                yield elem
