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

    def parse_tool_calls(self, content: str) -> tuple[str, list[ToolCall]]:
        """Given the string completion of the model, return the content without tool calls and the parsed tool calls."""
        try:
            data = json.loads(content)
            match data:
                case {"name": str(name), "parameters": dict(parameters)}:
                    tc = ToolCall.from_function_call(FunctionCall.with_args(name, **parameters))
                    return "", [tc]
        except json.JSONDecodeError:
            return content, []
        return content, []
