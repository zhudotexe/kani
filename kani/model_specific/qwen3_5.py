import logging
import re

from kani.models import FunctionCall, ToolCall

from .base import BaseParser
from ..engines.huggingface import ChatTemplatePromptPipeline

log = logging.getLogger(__name__)


# ===== PROMPT PIPELINE =====
def build_prompt_pipeline(tokenizer, **kwargs):
    # set default chat_template_reasoning_content_key to "reasoning_content"
    kwargs["chat_template_reasoning_content_key"] = (
        kwargs.get("chat_template_reasoning_content_key") or "reasoning_content"
    )
    return ChatTemplatePromptPipeline(tokenizer, **kwargs)


# ===== OUTPUT PARSER =====
class Qwen3_5Parser(BaseParser):
    r"""
    Tool calling + reasoning adapter for Qwen3 models::

        Qwen/Qwen3.5-*

    Reasoning segments are returned as :class:`.ReasoningPart`\ s.
    """

    def __init__(
        self,
        *args,
        tool_call_start_token="<tool_call>",
        tool_call_end_token="</tool_call>",
        reasoning_start_token="<think>",
        reasoning_end_token="</think>",
        reasoning_always_at_start=True,
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

    def parse_one_tool_call(self, content: str) -> ToolCall | None:
        """Parse one tool call from the <function=name>...</function> content. Return None if invalid."""
        # assumption: function names are valid Python identifiers
        function_match = re.match(r"<function=(\w+)>(.+?)</function>", content, re.DOTALL)
        if not function_match:
            log.warning(f"No <function> found when parsing tool call:\n{content}")
            return None

        tool_name = function_match[1]
        tool_args = {}
        param_matches = re.finditer(r"<parameter=(\w+)>\n?(.+?)\n?</parameter>", function_match[2], re.DOTALL)
        for param_match in param_matches:
            param_name = param_match[1]
            param_value = param_match[2]
            if param_name in tool_args:
                log.warning(f"Duplicate tool parameter found: {param_name!r} while parsing:\n{content}")
            tool_args[param_name] = param_value

        return ToolCall.from_function_call(FunctionCall.with_args(tool_name, **tool_args))

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
            tool_call = self.parse_one_tool_call(tool_content_match.group(1).strip())
            if tool_call is None:
                continue
            tool_calls.append(tool_call)
            content = content[: tool_content_match.start()] + content[tool_content_match.end() :]
        tool_calls.reverse()  # since we parsed backwards
        return content.strip(), tool_calls
