import json
import logging
import re

from kani.models import FunctionCall, ToolCall
from .base import BaseToolCallParser

log = logging.getLogger(__name__)


class Qwen3Parser(BaseToolCallParser):
    r"""
    Tool calling + reasoning adapter for Qwen3 models::

        Qwen/Qwen3-*

    Reasoning segments are returned as :class:`.ReasoningPart`\ s.
    """

    def __init__(
        self,
        *args,
        tool_call_start_token="<tool_call>",
        tool_call_end_token="</tool_call>",
        reasoning_start_token="<think>",
        reasoning_end_token="</think>",
        reasoning_always_at_start=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            tool_call_start_token=tool_call_start_token,
            tool_call_end_token=tool_call_end_token,
            reasoning_start_token=reasoning_start_token,
            reasoning_end_token=reasoning_end_token,
            reasoning_always_at_start=reasoning_always_at_start,
            **kwargs,
        )

    def parse_tool_calls(self, content: str):
        tool_calls = []
        tool_content_matches = re.finditer(
            rf"{re.escape(self.tool_call_start_token)}(.+?){re.escape(self.tool_call_end_token)}",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        # do this in reverse so we can edit the str directly
        for tool_content_match in reversed(list(tool_content_matches)):
            log.debug(f"Found tool content while parsing: {tool_content_match.group(1)}")
            try:
                data = json.loads(tool_content_match[1])
            except json.JSONDecodeError:
                log.error(
                    f"Could not decode tool content! Skipping this tool call:\n{tool_content_match[0]!r}!",
                    exc_info=True,
                )
                continue
            tool_name = data["name"]
            tool_args = data.get("arguments", {})
            tool_call = ToolCall.from_function_call(FunctionCall.with_args(tool_name, **tool_args))
            tool_calls.append(tool_call)
            content = content[: tool_content_match.start()] + content[tool_content_match.end() :]
        tool_calls.reverse()  # since we parsed backwards
        return content.strip(), tool_calls


class Qwen3ThinkingParser(Qwen3Parser):
    def __init__(self, *args, reasoning_always_at_start=True, **kwargs):
        super().__init__(*args, reasoning_always_at_start=reasoning_always_at_start, **kwargs)
