"""Model-agnostic classes used to represent the chat state and function calls."""

import abc
import enum
import json
import uuid
import warnings
from functools import cached_property
from typing import Any, ClassVar, Sequence, Type, TypeAlias, Union

from pydantic import BaseModel as PydanticBase, field_serializer, model_serializer, model_validator

from .exceptions import MissingMessagePartType

# ==== constants ====
MESSAGEPART_TYPE_KEY = "__kani_messagepart_type__"  # used for serdes of MessageParts

# ==== typing ====
MessagePartType: TypeAlias = Union["MessagePart", str]  # ChatMessage.parts[*]
QueryType: TypeAlias = str | Sequence[MessagePartType] | None  # Kani.*_round(...)


class BaseModel(PydanticBase, abc.ABC):
    """The base class for all Kani models."""

    def copy_with(self, **new_values):
        """Make a shallow copy of this object, updating the passed attributes (if any) to new values.
        This does not validate the updated attributes!
        This is mostly just a convenience wrapper around ``.model_copy``.
        """
        return self.model_copy(update=new_values)


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
    """Represents a model's request to call a single function."""

    name: str
    """The name of the requested function."""

    arguments: str
    """The arguments to call it with, encoded in JSON."""

    @cached_property
    def kwargs(self) -> dict[str, Any]:
        """The arguments to call the function with, as a Python dictionary."""
        return json.loads(self.arguments)

    @classmethod
    def with_args(cls, name: str, **kwargs):
        """Create a function call with the given arguments (e.g. for few-shot prompting)."""
        inst = cls(name=name, arguments=json.dumps(kwargs))
        inst.__dict__["kwargs"] = kwargs  # set the cached property here as a minor optimization
        return inst


class ToolCall(BaseModel):
    """Represents a model's request to call a tool with a unique request ID.

    See :ref:`functioncall_v_toolcall` for more information about tool calls vs function calls.
    """

    id: str
    """The request ID created by the engine.
    This should be passed back to the engine in :attr:`.ChatMessage.tool_call_id` in order to associate a FUNCTION
    message with this request.
    """

    type: str
    """The type of tool requested (currently only "function")."""

    function: FunctionCall
    """The requested function call."""

    @classmethod
    def from_function(cls, name: str, *, call_id_: str = None, **kwargs):
        """Create a tool call request for a function with the given name and arguments.

        :param call_id_: The ID to assign to the request. If not passed, generates a random ID.
        """
        call_id = call_id_ or str(uuid.uuid4())
        return cls(id=call_id, type="function", function=FunctionCall.with_args(name, **kwargs))

    @classmethod
    def from_function_call(cls, call: FunctionCall, call_id_: str = None):
        """Create a tool call request from an existing FunctionCall.

        :param call_id_: The ID to assign to the request. If not passed, generates a random ID.
        """
        call_id = call_id_ or str(uuid.uuid4())
        return cls(id=call_id, type="function", function=call)


class MessagePart(BaseModel, abc.ABC):
    """Base class for a part of a message.
    Engines should inherit from this class to tag substrings with metadata or provide multimodality to an engine.
    By default, if coerced to a string, will raise a warning noting that rich message part data was lost.
    For more information see :doc:`advanced/messageparts`.
    """

    # ==== serdes ====
    # used for saving/loading - map qualname to messagepart type
    _messagepart_registry: ClassVar[dict[str, Type["MessagePart"]]] = {}

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, **kwargs):
        """
        When a new MessagePart is defined, we need to save its type so that we can load saved JSON into the right type
        later.
        """
        super().__init_subclass__(**kwargs)
        fqn = cls.__module__ + "." + cls.__qualname__
        if fqn in cls._messagepart_registry:
            warnings.warn(
                f"The MessagePart type {fqn!r} was defined multiple times (perhaps a class is being defined in a"
                " function scope). This may cause issues when saving/loading messages with parts of this type."
            )
        cls._messagepart_registry[fqn] = cls

    @model_serializer(mode="wrap")
    def _serialize(self, nxt):
        """Extend the default serialization dict with a key recording what type it is."""
        retval = nxt(self)
        cls = type(self)
        fqn = cls.__module__ + "." + cls.__qualname__
        retval[MESSAGEPART_TYPE_KEY] = fqn
        return retval

    # noinspection PyNestedDecorators
    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, v, nxt):
        """If we are deserializing a dict with the special key, switch to the right class' validator."""
        if isinstance(v, dict) and MESSAGEPART_TYPE_KEY in v:
            fqn = v.pop(MESSAGEPART_TYPE_KEY)
            try:
                klass = cls._messagepart_registry[fqn]
            except KeyError:
                raise MissingMessagePartType(
                    fqn,
                    f"Found a MessagePart with type {fqn!r}, but the type is not defined. Maybe the type is from an"
                    " extension that has not yet been imported?",
                )
            return klass.model_validate(v)
        return nxt(v)

    # ==== entrypoints ====
    def __str__(self):
        """
        Used to define the fallback behaviour when a part is serialized to a string (e.g. via
        :attr:`.ChatMessage.text` ).
        Override this to specify the canonical string representation of your message part.

        Engines that support message parts should generally not use this, preferring to iterate over
        :attr:`.ChatMessage.parts` instead.
        """
        type_name = type(self).__name__
        warnings.warn(
            f"Message part of type {type_name!r} was coerced into a string. Rich data may not be visible to the"
            " user/model.\nDevelopers: If this warning is incorrect, please add support for this message part in your"
            f" engine or override `{type_name}.__str__()`."
        )
        return f"<{type_name} {super().__str__()}>"


class ChatMessage(BaseModel):
    """Represents a message in the chat context."""

    def __init__(self, **kwargs):
        # translate a function_call into tool_calls
        if "function_call" in kwargs:
            if "tool_calls" in kwargs:
                raise ValueError("Only one of `function_call` or `tool_calls` may be provided.")
            kwargs["tool_calls"] = (ToolCall.from_function_call(kwargs.pop("function_call")),)
        super().__init__(**kwargs)

    role: ChatRole
    """Who said the message?"""

    # ==== content ====
    content: str | list[MessagePart | str] | None
    """The data used to create this message. Generally, you should use :attr:`text` or :attr:`parts` instead."""

    @property
    def text(self) -> str | None:
        """The content of the message, as a string.
        Can be None only if the message is a requested function call from the assistant.
        If the message is comprised of multiple parts, concatenates the parts.
        """
        content = self.content
        if content is None:
            return None
        elif isinstance(content, str):
            return content
        return "".join(map(str, content))

    @property
    def parts(self) -> list[MessagePart | str]:
        """The parts of the message that make up its content.
        Can be an empty tuple only if the message is a requested function call from the assistant.

        This is a read-only list; changes here will not affect the message's content. To mutate the message content,
        use :meth:`copy_with` and set ``text``, ``parts``, or ``content``.
        """
        content = self.content
        if content is None:
            return []
        elif isinstance(content, str):
            return [content]
        return content

    name: str | None = None
    """The name of the user who sent the message, if set (user/function messages only)."""

    # ==== tool calling ====
    tool_call_id: str | None = None
    """The ID for a requested :class:`.ToolCall` which this message is a response to (function messages only)."""

    tool_calls: list[ToolCall] | None = None
    """The tool calls requested by the model (assistant messages only)."""

    is_tool_call_error: bool | None = None
    """If this is a FUNCTION message containing the results of a function call, whether the function call raised an
    exception."""

    @property
    def function_call(self) -> FunctionCall | None:
        """If there is exactly one tool call to a function, return that tool call's requested function.

        This is mostly provided for backwards-compatibility purposes; iterating over :attr:`tool_calls` should be
        preferred.
        """
        if not self.tool_calls:
            return None
        if len(self.tool_calls) > 1:
            warnings.warn(
                "This message contains multiple tool calls; iterate over `.tool_calls` instead of using"
                " `.function_call`."
            )
        return self.tool_calls[0].function

    # ==== constructors ====
    @classmethod
    def system(cls, content: str | Sequence[MessagePart | str], **kwargs):
        """Create a new system message."""
        return cls(role=ChatRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str | Sequence[MessagePart | str], **kwargs):
        """Create a new user message."""
        return cls(role=ChatRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(cls, content: str | Sequence[MessagePart | str] | None, **kwargs):
        """Create a new assistant message."""
        return cls(role=ChatRole.ASSISTANT, content=content, **kwargs)

    @classmethod
    def function(cls, name: str | None, content: str | Sequence[MessagePart | str], tool_call_id: str = None, **kwargs):
        """Create a new function message."""
        return cls(role=ChatRole.FUNCTION, content=content, name=name, tool_call_id=tool_call_id, **kwargs)

    # ==== helpers ====
    def copy_with(self, **new_values):
        """Make a shallow copy of this object, updating the passed attributes (if any) to new values.

        This does not validate the updated attributes!
        This is mostly just a convenience wrapper around ``.model_copy``.

        Only one of (content, text, parts) may be passed and will update the other two attributes accordingly.

        Only one of (tool_calls, function_call) may be passed and will update the other accordingly.
        """
        # === content ===
        # ensure that setting either text or parts works
        if "text" in new_values:
            if "content" in new_values:
                raise ValueError("At most one of ('content', 'text', 'parts') can be set.")
            new_values["content"] = new_values.pop("text")
        if "parts" in new_values:
            if "content" in new_values:
                raise ValueError("At most one of ('content', 'text', 'parts') can be set.")
            new_values["content"] = list(new_values.pop("parts"))

        # === tool calls ===
        if "function_call" in new_values:
            if "tool_calls" in new_values:
                raise ValueError("Only one of 'function_call' or 'tool_calls' may be provided.")
            new_values["tool_calls"] = (ToolCall.from_function_call(new_values.pop("function_call")),)

        return super().copy_with(**new_values)

    # ==== pydantic stuff ====
    @field_serializer("content", mode="wrap")
    def _content_serializer(self, content, nxt):
        """
        Custom serialization logic for a list of MessageParts due to
        https://docs.pydantic.dev/latest/concepts/serialization/#subclass-instances-for-fields-of-basemodel-dataclasses-typeddict
        """
        if not isinstance(content, list):
            return nxt(content)

        out = []
        for item in content:
            if isinstance(item, MessagePart):
                out.append(item.model_dump())
            else:
                out.append(nxt(item))
        return out
