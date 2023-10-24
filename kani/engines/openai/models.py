from typing import Literal

from kani.models import BaseModel, ChatMessage, ChatRole, FunctionCall
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


class SpecificFunctionCall(BaseModel):
    name: str


class OpenAIChatMessage(BaseModel):
    role: ChatRole
    content: str | list[BaseModel | str] | None
    name: str | None = None
    function_call: FunctionCall | None = None

    @classmethod
    def from_chatmessage(cls, m: ChatMessage):
        return cls(role=m.role, content=m.text, name=m.name, function_call=m.function_call)


# ---- response ----
class ChatCompletionChoice(BaseModel):
    # this is a ChatMessage rather than an OpenAIChatMessage because all engines need to return the kani model
    message: ChatMessage
    index: int
    finish_reason: str | None = None


class ChatCompletion(BaseCompletion, BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    usage: CompletionUsage
    choices: list[ChatCompletionChoice]

    @property
    def message(self):
        return self.choices[0].message

    @property
    def prompt_tokens(self):
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self):
        # for some reason, the OpenAI API doesn't return the tokens used by ChatML
        # so we add on the length of "<|im_start|>assistant" and "<|im_end|>" here
        return self.usage.completion_tokens + 5
