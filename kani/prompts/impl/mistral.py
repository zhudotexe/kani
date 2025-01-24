import json
import logging

from kani.ai_function import AIFunction
from kani.models import ChatMessage, ChatRole, ToolCall
from kani.prompts import ApplyContext, PromptPipeline

log = logging.getLogger(__name__)


# ref: https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/sentencepiece.py
def maybe_json(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def json_tool_call(tc: ToolCall):
    tc_json = {"name": tc.function.name, "arguments": tc.function.kwargs, "id": tc.id}
    return json.dumps(tc_json)


def fmt_function_call_result(msg: ChatMessage):
    result_json = {"content": maybe_json(msg.text), "call_id": msg.tool_call_id}
    msg.content = json.dumps(result_json)
    return msg


def _fmt_functions(functions: list[AIFunction]) -> str:
    tools_payload = [
        {
            "type": "function",
            "function": {"name": f.name, "description": f.desc, "parameters": f.json_schema},
        }
        for f in functions
    ]
    tools_json = json.dumps(tools_payload)
    return f"[AVAILABLE_TOOLS] {tools_json}[/AVAILABLE_TOOLS]"


def fmt_available_tools(msg: ChatMessage, ctx: ApplyContext) -> ChatMessage:
    # prepend tools on the last user message
    if ctx.functions and ctx.is_last_of_type:
        msg.content = f"{_fmt_functions(ctx.functions)}{msg.text}"
    return msg


def ensure_available_tools(msgs: list[ChatMessage], functions: list[AIFunction]) -> list[ChatMessage]:
    if functions and not msgs:
        msgs.insert(0, ChatMessage.user(_fmt_functions(functions)))
    return msgs


MISTRAL_V3_PIPELINE = (
    PromptPipeline()
    # Mistral does not support SYSTEM messages - translate with a warning
    .translate_role(
        role=ChatRole.SYSTEM,
        to=ChatRole.USER,
        warn=(
            "The Mistral prompt format does not natively support the SYSTEM role. These messages will be"
            " sent to the model as USER messages."
        ),
    )
    # If we see two consecutive USER messages, merge them together into one with a
    # newline in between.
    .merge_consecutive(role=ChatRole.USER, sep="\n\n")
    # Similarly for ASSISTANT, but with a space (kani automatically strips whitespace from the ends of
    # generations).
    .merge_consecutive(role=ChatRole.ASSISTANT, sep=" ")
    # We wrap USER messages here since we do some shenanigans in the next step
    .wrap(role=ChatRole.USER, prefix="[INST] ", suffix="[/INST]")
    # --- function calling ---
    .ensure_bound_function_calls(id_translator=lambda x: x.replace("-", "")[:9])
    # Format function calls with the [TOOL_CALLS] format.
    .function_call_fmt(json_tool_call, prefix="[TOOL_CALLS] [", sep=",", suffix="]")
    # Include the call ID in the FUNCTION result.
    .apply(fmt_function_call_result, role=ChatRole.FUNCTION)
    # Include the list of available functions just before the last user message
    .apply(fmt_available_tools, role=ChatRole.USER)
    # (or prepend if none)
    .macro_apply(ensure_available_tools)
    # --- output ---
    # Finally, wrap USER and ASSISTANT messages in the instruction tokens. If our
    # message list ends with an ASSISTANT message, don't add the EOS token
    # (we want the model to continue the generation).
    .conversation_fmt(
        prefix="<s>",
        assistant_prefix=" ",
        assistant_suffix="</s>",
        assistant_suffix_if_last="",
        function_prefix="[TOOL_RESULTS] ",
        function_suffix="[/TOOL_RESULTS]",
    )
)


# {%- if messages[0]["role"] == "system" %}
#     {%- set system_message = messages[0]["content"] %}
#     {%- set loop_messages = messages[1:] %}
# {%- else %}
#     {%- set loop_messages = messages %}
# {%- endif %}
# {%- if not tools is defined %}
#     {%- set tools = none %}
# {%- endif %}
# {%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}
# {#- This block checks for alternating user/assistant messages, skipping tool calling messages #}
# {%- set ns = namespace() %}
# {%- set ns.index = 0 %}
# {%- for message in loop_messages %}
#     {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}
#         {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}
#             {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}
#         {%- endif %}
#         {%- set ns.index = ns.index + 1 %}
#     {%- endif %}
# {%- endfor %}
# {{- bos_token }}
# {%- for message in loop_messages %}
#     {%- if message["role"] == "user" %}
#         {%- if tools is not none and (message == user_messages[-1]) %}
#             {{- "[AVAILABLE_TOOLS] [" }}
#             {%- for tool in tools %}
#                 {%- set tool = tool.function %}
#                 {{- '{"type": "function", "function": {' }}
#                 {%- for key, val in tool.items() if key != "return" %}
#                     {%- if val is string %}
#                         {{- '"' + key + '": "' + val + '"' }}
#                     {%- else %}
#                         {{- '"' + key + '": ' + val|tojson }}
#                     {%- endif %}
#                     {%- if not loop.last %}
#                         {{- ", " }}
#                     {%- endif %}
#                 {%- endfor %}
#                 {{- "}}" }}
#                 {%- if not loop.last %}
#                     {{- ", " }}
#                 {%- else %}
#                     {{- "]" }}
#                 {%- endif %}
#             {%- endfor %}
#             {{- "[/AVAILABLE_TOOLS]" }}
#             {%- endif %}
#         {%- if loop.last and system_message is defined %}
#             {{- "[INST] " + system_message + "\n\n" + message["content"] + "[/INST]" }}
#         {%- else %}
#             {{- "[INST] " + message["content"] + "[/INST]" }}
#         {%- endif %}
#     {%- elif message.tool_calls is defined and message.tool_calls is not none %}
#         {{- "[TOOL_CALLS] [" }}
#         {%- for tool_call in message.tool_calls %}
#             {%- set out = tool_call.function|tojson %}
#             {{- out[:-1] }}
#             {%- if not tool_call.id is defined or tool_call.id|length != 9 %}
#                 {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
#             {%- endif %}
#             {{- ', "id": "' + tool_call.id + '"}' }}
#             {%- if not loop.last %}
#                 {{- ", " }}
#             {%- else %}
#                 {{- "]" + eos_token }}
#             {%- endif %}
#         {%- endfor %}
#     {%- elif message["role"] == "assistant" %}
#         {{- " " + message["content"]|trim + eos_token}}
#     {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
#         {%- if message.content is defined and message.content.content is defined %}
#             {%- set content = message.content.content %}
#         {%- else %}
#             {%- set content = message.content %}
#         {%- endif %}
#         {{- '[TOOL_RESULTS] {"content": ' + content|string + ", " }}
#         {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
#             {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
#         {%- endif %}
#         {{- '"call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
#     {%- else %}
#         {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
#     {%- endif %}
# {%- endfor %}


# ==== function call parsing ====
# implemented in tool_adapters/mistral - here for back-compat
from kani.tool_parsers.mistral import MistralToolCallParser as MistralFunctionCallingAdapter  # noqa E402

MixtralFunctionCallingAdapter = MistralFunctionCallingAdapter
