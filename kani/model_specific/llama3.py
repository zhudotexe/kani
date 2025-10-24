"""Common builder for the LLaMAv3-chat prompt."""

import json

from kani.model_specific import BaseToolCallParser
from kani.models import ChatRole, FunctionCall, ToolCall
from kani.prompts.pipeline import PromptPipeline

# ===== Llama 3 =====
# llama 3 only; 3.1 and 3.2 are handled by the chat template in HF
LLAMA3_PIPELINE = (
    PromptPipeline()
    .translate_role(
        role=ChatRole.FUNCTION,
        to=ChatRole.USER,
        warn=(
            "The Llama 3 prompt format does not natively support the FUNCTION role. These messages will be"
            " sent to the model as USER messages."
        ),
    )
    .conversation_fmt(
        prefix="<|begin_of_text|>",
        generation_suffix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        user_suffix="<|eot_id|>",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_suffix="<|eot_id|>",
        assistant_suffix_if_last="",
        system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
        system_suffix="<|eot_id|>",
    )
)  # fmt: skip

# from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
# {% set loop_messages = messages %}
# {% for message in loop_messages %}
#   {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}
#   {% if loop.index0 == 0 %}
#       {% set content = bos_token + content %}
#   {% endif %}
#   {{ content }}
# {% endfor %}
# {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}


# ===== Llama 3.1, 3.2, 3.3 =====
# tool parser only, chat template handled by HF
class Llama3XToolCallParser(BaseToolCallParser):
    """
    Tool calling adapter for the Llama 3.x series of models which output JSON for tool calling.

    <|python_tag|>{"name": ..., "parameters": ...}
    """

    def __init__(self, *args, tool_call_start_token="<|python_tag|>", tool_call_end_token=None, **kwargs):
        super().__init__(
            *args, tool_call_start_token=tool_call_start_token, tool_call_end_token=tool_call_end_token, **kwargs
        )

    def parse_tool_calls(self, content: str) -> tuple[str, list[ToolCall]]:
        """Given the string completion of the model, return the content without tool calls and the parsed tool calls."""
        # sometimes starts with this
        content = content.removeprefix(self.tool_call_start_token)
        try:
            data = json.loads(content.strip())
            match data:
                case {"name": str(name), "parameters": dict(parameters)}:
                    tc = ToolCall.from_function_call(FunctionCall.with_args(name, **parameters))
                    return "", [tc]
                case {"name": str(name)}:
                    tc = ToolCall.from_function_call(FunctionCall.with_args(name))
                    return "", [tc]
        except json.JSONDecodeError:
            return content, []
        return content, []
