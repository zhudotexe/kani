from typing import Literal

from kani.models import BaseModel, ChatMessage, ChatRole, FunctionCall, ToolCall
from ..base import BaseCompletion


# ==== text completions ====
class CompletionLogProbs(BaseModel):
    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]]
    text_offset: list[int]


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: CompletionLogProbs | None = None
    finish_reason: str | None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Completion(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    system_fingerprint: str | None = None
    choices: list[CompletionChoice]
    usage: CompletionUsage

    @property
    def text(self):
        """The text of the top completion."""
        return self.choices[0].text


# ==== chat completions ====
class FunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict


class ToolSpec(BaseModel):
    type: str
    function: FunctionSpec

    @classmethod
    def from_function(cls, spec: FunctionSpec):
        return cls(type="function", function=spec)


class SpecificFunctionCall(BaseModel):
    name: str


class ToolChoice(BaseModel):
    type: str
    function: SpecificFunctionCall

    @classmethod
    def from_function(cls, name: str):
        return cls(type="function", function=SpecificFunctionCall(name=name))


class ResponseFormat(BaseModel):
    type: str

    @classmethod
    def text(cls):
        return cls(type="text")

    @classmethod
    def json_object(cls):
        return cls(type="json_object")


class OpenAIChatMessage(BaseModel):
    role: str
    content: str | list[BaseModel | str] | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    # deprecated
    function_call: FunctionCall | None = None

    @classmethod
    def from_chatmessage(cls, m: ChatMessage):
        # translate tool responses to a function to the right openai format
        if m.role == ChatRole.FUNCTION:
            if m.tool_call_id is not None:
                return cls(role="tool", content=m.text, name=m.name, tool_call_id=m.tool_call_id)
            return cls(role=m.role.value, content=m.text, name=m.name)
        return cls(role=m.role.value, content=m.text, name=m.name, tool_call_id=m.tool_call_id, tool_calls=m.tool_calls)

    def to_chatmessage(self) -> ChatMessage:
        # translate tool role to function role
        if self.role == "tool":
            role = ChatRole.FUNCTION
        else:
            role = ChatRole(self.role)
        # translate FunctionCall to singular ToolCall
        if self.tool_calls:
            tool_calls = self.tool_calls
        elif self.function_call:
            tool_calls = [ToolCall.from_function_call(self.function_call)]
        else:
            tool_calls = None
        return ChatMessage(
            role=role, content=self.content, name=self.name, tool_call_id=self.tool_call_id, tool_calls=tool_calls
        )


# ---- response ----
class ChatCompletionChoice(BaseModel):
    message: OpenAIChatMessage
    index: int
    finish_reason: str | None = None


class ChatCompletion(BaseCompletion, BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    system_fingerprint: str | None = None
    usage: CompletionUsage
    choices: list[ChatCompletionChoice]

    @property
    def message(self) -> ChatMessage:
        return self.choices[0].message.to_chatmessage()

    @property
    def prompt_tokens(self):
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self):
        # for some reason, the OpenAI API doesn't return the tokens used by ChatML
        # so we add on the length of "<|im_start|>assistant" and "<|im_end|>" here
        return self.usage.completion_tokens + 5
