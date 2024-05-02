import json
import re

from kani.ai_function import AIFunction
from kani.engines import Completion, WrapperEngine
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall
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


# ==== function call parsing ====
# [TOOL_CALLS][{'name': 'get_current_weather', 'arguments': {'location': 'Paris, France', 'format': 'celsius'}}]</s>
class MixtralFunctionCallingAdapter(WrapperEngine):
    """Common Mixtral-8x22B function calling parsing wrapper."""

    @staticmethod
    def _parse_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
        tool_json = re.search(r"\[TOOL_CALLS](.+)</s>", content, re.IGNORECASE | re.DOTALL)
        if tool_json is None:
            return content, []
        actions = json.loads(tool_json.group(1))

        # translate back to kani spec
        tool_calls = []
        for action in actions:
            tool_name = action["name"]
            tool_args = json.dumps(action["arguments"])
            tool_id = action.get("id")
            tool_call = ToolCall.from_function_call(FunctionCall(name=tool_name, arguments=tool_args), call_id_=tool_id)
            tool_calls.append(tool_call)

        # return trimmed content and tool calls
        return content[: tool_json.start()], tool_calls

    async def predict(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams):
        completion = await super().predict(messages, functions, **hyperparams)

        # if we have tools, parse
        if functions:
            completion.message.content, completion.message.tool_calls = self._parse_tool_calls(completion.message.text)

        return completion

    async def stream(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams):
        content_parts = []
        in_tool_call = False
        inner_completion = None

        # consume from the inner iterator, yielding as normal until we see a tool call or a completion
        async for elem in super().stream(messages, functions, **hyperparams):
            if isinstance(elem, str):
                content_parts.append(elem)
                # if we see the start of a tool call, stop yielding and start buffering
                if elem == "[TOOL_CALLS]":
                    in_tool_call = True
                # otherwise yield the string
                if not in_tool_call:
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
            content, tool_calls = self._parse_tool_calls(content)
            if inner_completion:
                tool_calls = (inner_completion.message.tool_calls or []) + tool_calls
            yield Completion(ChatMessage.assistant(content, tool_calls=tool_calls))
