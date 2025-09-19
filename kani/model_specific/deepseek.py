import json
import logging
import re

from kani.models import FunctionCall, ToolCall
from .base import BaseToolCallParser

log = logging.getLogger(__name__)


class DeepSeekR1ToolCallParser(BaseToolCallParser):
    """
    Tool calling adapter for DeepSeek models using the R1 tool call format::

        deepseek-ai/DeepSeek-R1
        deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
        deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
        deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
        deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
        deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    """

    def __init__(
        self,
        *args,
        tool_call_start_token: str = "<｜tool▁calls▁begin｜>",
        tool_call_end_token: str = "<｜tool▁outputs▁end｜>",
        **kwargs,
    ):
        super().__init__(
            *args, tool_call_start_token=tool_call_start_token, tool_call_end_token=tool_call_end_token, **kwargs
        )

    def parse_tool_calls(self, content: str):
        tool_content_match = re.search(
            rf"{re.escape(self.tool_call_start_token)}\s*(.+?)\s*({re.escape(self.tool_call_end_token)})",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if tool_content_match is None:
            return content, []
        log.debug(f"Found tool content while parsing: {tool_content_match.group(1)}")

        # translate to kani spec
        tool_calls = []
        for tc_match in re.finditer(
            r"<｜tool▁call▁begin｜>(?P<type>.+?)<｜tool▁sep｜>(?P<name>.+?)\n```json\n(?P<args>.+?)\n```<｜tool▁call▁end｜>",
            tool_content_match.group(1),
            re.IGNORECASE | re.DOTALL,
        ):
            tool_name = tc_match["name"].strip()
            tool_args = tc_match["args"].strip()
            try:
                json.loads(tool_args)
            except json.JSONDecodeError:
                log.error(f"Could not decode tool content! Skipping this tool call:\n{tc_match[0]!r}!", exc_info=True)
                continue
            tool_call = ToolCall.from_function_call(FunctionCall(name=tool_name, arguments=tool_args))
            tool_calls.append(tool_call)

        # return trimmed content and tool calls
        return content[: tool_content_match.start()], tool_calls


# ===== deepseek-r1 function calling =====
# {% if not add_generation_prompt is defined %}
#   {% set add_generation_prompt = false %}
# {% endif %}
# {% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}
# {%- for message in messages %}
#   {%- if message['role'] == 'system' %}
#     {%- if ns.is_first_sp %}
#       {% set ns.system_prompt = ns.system_prompt + message['content'] %}
#       {% set ns.is_first_sp = false %}
#     {%- else %}
#       {% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}
#     {%- endif %}
#   {%- endif %}
# {%- endfor %}
# {{ bos_token }}{{ ns.system_prompt }}
# {%- for message in messages %}
#   {%- if message['role'] == 'user' %}
#     {%- set ns.is_tool = false -%}
#     {{'<｜User｜>' + message['content']}}
#   {%- endif %}
#   {%- if message['role'] == 'assistant' and 'tool_calls' in message %}
#     {%- set ns.is_tool = false -%}
#     {%- for tool in message['tool_calls'] %}
#       {%- if not ns.is_first %}
#         {%- if message['content'] is none %}
#           {{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}
#         {%- else %}
#           {{'<｜Assistant｜>' + message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}
#         {%- endif %}
#         {%- set ns.is_first = true -%}
#       {%- else %}
#         {{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}
#       {%- endif %}
#     {%- endfor %}
#   {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
#   {%- endif %}
#   {%- if message['role'] == 'assistant' and 'tool_calls' not in message %}
#     {%- if ns.is_tool %}
#       {{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}
#       {%- set ns.is_tool = false -%}
#     {%- else %}
#       {% set content = message['content'] %}
#       {% if '</think>' in content %}
#         {% set content = content.split('</think>')[-1] %}
#       {% endif %}
#       {{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}
#     {%- endif %}
#   {%- endif %}
#   {%- if message['role'] == 'tool' %}
#     {%- set ns.is_tool = true -%}
#     {%- if ns.is_output_first %}
#       {{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
#       {%- set ns.is_output_first = false %}
#     {%- else %}
#       {{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
#     {%- endif %}
#   {%- endif %}
# {%- endfor -%}
# {% if ns.is_tool %}
#   {{'<｜tool▁outputs▁end｜>'}}
# {% endif %}
# {% if add_generation_prompt and not ns.is_tool %}
#   {{'<｜Assistant｜>'}}
# {% endif %}
