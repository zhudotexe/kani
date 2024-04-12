from collections.abc import Collection
from typing import Callable, TypeVar

from kani.models import ChatMessage, ChatRole, MessagePart, ToolCall

PipelineMsgT = ChatMessage
"""The type of messages in the pipeline"""

MessageContentT = str | list[MessagePart | str] | None
"""The type of ChatMessage.content"""

RoleFilterT = ChatRole | Collection[ChatRole]
"""A role or list of roles to apply a step to"""

PredicateFilterT = Callable[[PipelineMsgT], bool]
"""A callable that determines whether or not to apply a step to an input"""

FunctionCallStrT = Callable[[ToolCall], str | None]
"""A callable to format a toolcall as a str"""

ApplyResultT = TypeVar("ApplyResultT")

ApplyCallableT = (
    Callable[[PipelineMsgT], ApplyResultT]
    | Callable[[PipelineMsgT, bool], ApplyResultT]
    | Callable[[PipelineMsgT, bool, int], ApplyResultT]
)
"""A function taking 1-3 args"""
