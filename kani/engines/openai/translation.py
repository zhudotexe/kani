"""Helpers to translate kani chat objects into OpenAI params."""

from kani.ai_function import AIFunction
from kani.engines.base import BaseCompletion
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall
from kani.prompts.pipeline import PromptPipeline

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
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
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


# main
OPENAI_PIPELINE = (
    PromptPipeline()
    .ensure_bound_function_calls()
    .ensure_start(predicate=lambda msg: msg.role != ChatRole.FUNCTION)
    .apply(kani_cm_to_openai_cm)
)


def translate_functions(functions: list[AIFunction]) -> list[ChatCompletionToolParam]:
    return [
        ChatCompletionToolParam(
            type="function", function=FunctionDefinition(name=f.name, description=f.desc, parameters=f.json_schema)
        )
        for f in functions
    ]


def translate_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    return OPENAI_PIPELINE(messages)


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


def openai_tc_to_kani_tc(tc: ChatCompletionMessageToolCall | ChoiceDeltaToolCall) -> ToolCall:
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
