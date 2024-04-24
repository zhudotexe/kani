from collections.abc import Collection
from dataclasses import dataclass
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


@dataclass
class ApplyContext:
    """Context about where a message lives in the pipeline for an arbitrary Apply operation."""

    msg: PipelineMsgT
    is_last: bool
    idx: int
    messages: list[PipelineMsgT]

    @property
    def is_last_of_type(self):
        """This message is the last one of its role in the pipeline."""
        return self.msg is [m for m in self.messages if m.role == self.msg.role][-1]


ApplyCallableT = Callable[[PipelineMsgT], ApplyResultT] | Callable[[PipelineMsgT, ApplyContext], ApplyResultT]
"""A function taking 1-2 args"""
