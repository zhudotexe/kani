import json

from kani.ai_function import AIFunction
from kani.models import ChatMessage, ChatRole, ToolCall
from kani.prompts import ApplyContext, PromptPipeline


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
    result_json = {"call_id": msg.tool_call_id, "content": maybe_json(msg.text)}
    msg.content = json.dumps(result_json)
    return msg


def _fmt_functions(functions: list[AIFunction]) -> str:
    tools_json = [
        {
            "type": "function",
            "function": {"name": f.name, "description": f.desc, "parameters": f.json_schema},
        }
        for f in functions
    ]
    return f"[AVAILABLE_TOOLS]{tools_json}[/AVAILABLE_TOOLS]"


def fmt_available_tools(msg: ChatMessage, ctx: ApplyContext) -> ChatMessage:
    # prepend tools on the last user message
    if ctx.is_last_of_type:
        msg.content = f"{_fmt_functions(ctx.functions)}{msg.text}"
    return msg


def ensure_available_tools(msgs: list[ChatMessage], functions: list[AIFunction]) -> list[ChatMessage]:
    if not msgs:
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
    .wrap(role=ChatRole.USER, prefix="[INST]", suffix="[/INST]")
    # --- function calling ---
    .ensure_bound_function_calls()
    # Format function calls with the [TOOL_CALLS] format.
    .function_call_fmt(json_tool_call, prefix="[TOOL_CALLS][", sep=",", suffix="]</s>")
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
        assistant_suffix=" </s>",
        assistant_suffix_if_last="",
        function_prefix="[TOOL_RESULTS]",
        function_suffix="[/TOOL_RESULTS]",
    )
)

# tool use template
# {{bos_token}}
# {% set user_messages = messages | selectattr('role', 'equalto', 'user') | list %}
# {% for message in messages %}
#   {% if message['role'] == 'user' %}
#       {% if message == user_messages[-1] %}
#           {% if tools %}
#               {{'[AVAILABLE_TOOLS]'+ tools|string + '[/AVAILABLE_TOOLS]'}}
#           {% endif %}
#           {{ '[INST]' + message['content'] + '[/INST]' }}
#       {% else %}
#           {{ '[INST]' + message['content'] + '[/INST]' }}
#       {% endif %}
#   {% elif message['role'] == 'assistant' %}
#       {{ ' ' + message['content'] + ' ' + eos_token}}
#   {% elif message['role'] == 'tool_results' %}
#       {{'[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]'}}
#   {% elif message['role'] == 'tool_calls' %}
#       {{'[TOOL_CALLS]' + message['content']|string + eos_token}}
#   {% endif %}
# {% endfor %}"
