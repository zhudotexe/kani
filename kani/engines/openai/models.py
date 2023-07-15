from typing import Literal

from pydantic import BaseModel

from kani.models import ChatMessage
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


# ---- response ----
class ChatCompletionChoice(BaseModel):
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

    @property
    def text(self):
        """The text of the most recent chat completion."""
        return self.message.content
