"""Model-agnostic classes used to represent the chat state and function calls."""
import enum
import json

from pydantic import BaseModel, ConfigDict


# ==== chat ====
class ChatRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class FunctionCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    arguments: str

    @property
    def kwargs(self):
        return json.loads(self.arguments)


class ChatMessage(BaseModel):
    model_config = ConfigDict(use_enum_values=True, frozen=True)

    role: ChatRole
    content: str | None
    name: str | None = None
    function_call: FunctionCall | None = None

    @classmethod
    def system(cls, content):
        return cls(role=ChatRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content):
        return cls(role=ChatRole.USER, content=content)

    @classmethod
    def assistant(cls, content):
        return cls(role=ChatRole.ASSISTANT, content=content)

    @classmethod
    def function(cls, name, content):
        return cls(role=ChatRole.FUNCTION, content=content, name=name)
