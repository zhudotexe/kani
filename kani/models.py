"""Model-agnostic classes used to represent the chat state and function calls."""
import enum
import json

from pydantic import BaseModel, ConfigDict


# ==== chat ====
class ChatRole(enum.Enum):
    """Represents who said a chat message."""

    SYSTEM = "system"
    """The message is from the system (usually a steering prompt)."""

    USER = "user"
    """The message is from the user."""

    ASSISTANT = "assistant"
    """The message is from the language model."""

    FUNCTION = "function"
    """The message is the result of a function call."""


class FunctionCall(BaseModel):
    """Represents a model's request to call a function."""

    model_config = ConfigDict(frozen=True)

    name: str
    """The name of the requested function."""

    arguments: str
    """The arguments to call it with, encoded in JSON."""

    @property
    def kwargs(self) -> dict:
        """The arguments to call the function with, with JSON decoded to a Python dict."""
        return json.loads(self.arguments)

    @classmethod
    def with_args(cls, name: str, **kwargs):
        """Create a function call with the given arguments (e.g. for few-shot prompting)."""
        return cls(name=name, arguments=json.dumps(kwargs))


class ChatMessage(BaseModel):
    """Represents a message in the chat context."""

    model_config = ConfigDict(frozen=True)

    role: ChatRole
    """Who said the message?"""

    content: str | None
    """The content of the message. Can be None only if the message is a requested function call from the assistant."""

    name: str | None = None
    """The name of the user who sent the message, if set (user messages only)."""

    function_call: FunctionCall | None = None
    """The function requested by the model (assistant messages only)."""

    @classmethod
    def system(cls, content: str, **kwargs):
        """Create a new system message."""
        return cls(role=ChatRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs):
        """Create a new user message."""
        return cls(role=ChatRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str | None, **kwargs):
        """Create a new assistant message."""
        return cls(role=ChatRole.ASSISTANT, content=content, **kwargs)

    @classmethod
    def function(cls, name: str, content: str, **kwargs):
        """Create a new function message."""
        return cls(role=ChatRole.FUNCTION, content=content, name=name, **kwargs)
