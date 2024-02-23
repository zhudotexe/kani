"""Helpers to translate kani chat objects into OpenAI params."""

from kani.ai_function import AIFunction
from kani.engines.base import BaseCompletion
from kani.exceptions import MissingModelDependencies, PromptError
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall

try:
    from openai.types.chat import (
        ChatCompletion as OpenAIChatCompletion,
        ChatCompletionAssistantMessageParam,
        ChatCompletionFunctionMessageParam,
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCall,
        ChatCompletionMessageToolCallParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionToolParam,
        ChatCompletionUserMessageParam,
    )
    from openai.types.chat.chat_completion_message_tool_call import Function as ChatCompletionMessageFunctionCall
    from openai.types.chat.chat_completion_message_tool_call_param import (
        Function as ChatCompletionMessageToolCallFunctionParam,
    )
    from openai.types.shared_params import FunctionDefinition
except ImportError as e:
    raise MissingModelDependencies(
        'The OpenAIEngine requires extra dependencies. Please install kani with "pip install kani[openai]".'
    ) from None


# ==== kani -> openai ====
def translate_functions(functions: list[AIFunction]) -> list[ChatCompletionToolParam]:
    return [
        ChatCompletionToolParam(
            type="function", function=FunctionDefinition(name=f.name, description=f.desc, parameters=f.json_schema)
        )
        for f in functions
    ]


def translate_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    translated_messages = []
    free_toolcall_ids = set()
    for m in messages:
        # if this is not a function result and there are free tool call IDs, raise
        if m.role != ChatRole.FUNCTION and free_toolcall_ids:
            raise PromptError(
                f"Encountered a {m.role.value!r} message but expected a FUNCTION message to satisfy the pending"
                f" tool call(s): {free_toolcall_ids}"
            )
        # asst: add tool call IDs to freevars
        if m.role == ChatRole.ASSISTANT and m.tool_calls:
            for tc in m.tool_calls:
                free_toolcall_ids.add(tc.id)
        # func: bind freevars
        elif m.role == ChatRole.FUNCTION:
            # has ID: bind it if requested; translate to FUNCTION if not
            if m.tool_call_id is not None:
                if m.tool_call_id in free_toolcall_ids:
                    free_toolcall_ids.remove(m.tool_call_id)
                else:
                    # this happens if the tool call is pushed out of context but the result is still here,
                    # and we have always included messages beforehand
                    # TODO: this will eventually be deprecated - maube we just skip this message?
                    m = m.copy_with(tool_call_id=None)
            # no ID: bind if unambiguous
            elif len(free_toolcall_ids) == 1:
                m = m.copy_with(tool_call_id=free_toolcall_ids.pop())
            # no ID: error if ambiguous
            elif len(free_toolcall_ids) > 1:
                raise PromptError(
                    "Got a FUNCTION message with no tool_call_id but multiple tool calls are pending"
                    f" ({free_toolcall_ids})! Set the tool_call_id to resolve the pending tool requests."
                )
            # otherwise pass the FUNCTION message through
        translated_messages.append(kani_cm_to_openai_cm(m))
    # if the translated messages start with a hanging TOOL call, strip it (openai limitation)
    # though hanging FUNCTION messages are OK
    while translated_messages and translated_messages[0]["role"] == "tool":
        translated_messages.pop(0)
    return translated_messages


# decomp
def kani_cm_to_openai_cm(msg: ChatMessage) -> ChatCompletionMessageParam:
    """Translate a kani ChatMessage into an OpenAI Message."""
    # translate tool responses to a function to the right openai format
    match msg.role:
        case ChatRole.FUNCTION if msg.tool_call_id is not None:
            return ChatCompletionToolMessageParam(role="tool", content=msg.text, tool_call_id=msg.tool_call_id)
        case ChatRole.FUNCTION:
            return ChatCompletionFunctionMessageParam(**_msg_kwargs(msg))
        case ChatRole.SYSTEM:
            return ChatCompletionSystemMessageParam(**_msg_kwargs(msg))
        case ChatRole.USER:
            return ChatCompletionUserMessageParam(**_msg_kwargs(msg))
        case _:  # assistant
            if msg.tool_calls:
                tool_calls = [kani_tc_to_openai_tc(tc) for tc in msg.tool_calls]
                return ChatCompletionAssistantMessageParam(**_msg_kwargs(msg), tool_calls=tool_calls)
            return ChatCompletionAssistantMessageParam(**_msg_kwargs(msg))


def _msg_kwargs(msg: ChatMessage) -> dict:
    data = dict(role=msg.role.value, content=msg.text)
    if msg.name is not None:
        data["name"] = msg.name
    return data


def kani_tc_to_openai_tc(tc: ToolCall) -> ChatCompletionMessageToolCallParam:
    """Translate a kani ToolCall into an OpenAI dict"""
    oai_function = ChatCompletionMessageToolCallFunctionParam(name=tc.function.name, arguments=tc.function.arguments)
    return ChatCompletionMessageToolCallParam(id=tc.id, type="function", function=oai_function)


# ==== openai -> kani ====
def openai_cm_to_kani_cm(msg: ChatCompletionMessage) -> ChatMessage:
    """Translate an OpenAI ChatCompletionMessage into a kani ChatMessage."""
    # translate tool role to function role
    if msg.role == "tool":
        role = ChatRole.FUNCTION
    else:
        role = ChatRole(msg.role)
    # translate FunctionCall to singular ToolCall
    if msg.tool_calls:
        tool_calls = [openai_tc_to_kani_tc(tc) for tc in msg.tool_calls]
    elif msg.function_call:
        tool_calls = [ToolCall.from_function_call(msg.function_call)]
    else:
        tool_calls = None
    return ChatMessage(role=role, content=msg.content, tool_calls=tool_calls)


def openai_tc_to_kani_tc(tc: ChatCompletionMessageToolCall) -> ToolCall:
    return ToolCall(id=tc.id, type=tc.type, function=openai_fc_to_kani_fc(tc.function))


def openai_fc_to_kani_fc(fc: ChatCompletionMessageFunctionCall) -> FunctionCall:
    return FunctionCall(name=fc.name, arguments=fc.arguments)


class ChatCompletion(BaseCompletion):
    """A wrapper around the OpenAI ChatCompletion to make it compatible with the Kani interface."""

    def __init__(self, openai_completion: OpenAIChatCompletion):
        self.openai_completion = openai_completion
        """The underlying OpenAI ChatCompletion."""
        self._message = openai_cm_to_kani_cm(openai_completion.choices[0].message)

    @property
    def message(self):
        return self._message

    @property
    def prompt_tokens(self):
        return self.openai_completion.usage.prompt_tokens

    @property
    def completion_tokens(self):
        # for some reason, the OpenAI API doesn't return the tokens used by ChatML
        # so we add on the length of "<|im_start|>assistant" and "<|im_end|>" here
        return self.openai_completion.usage.completion_tokens + 5
