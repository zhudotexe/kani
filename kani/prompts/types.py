from collections.abc import Collection
from typing import Any, Callable

from kani.models import ChatMessage, ChatRole, ToolCall

PipelineMsgT = ChatMessage
"""The type of messages in the pipeline"""

RoleFilterT = ChatRole | Collection[ChatRole]
"""A role or list of roles to apply a step to"""

PredicateFilterT = Callable[[PipelineMsgT], bool]
"""A callable that determines whether or not to apply a step to an input"""

FunctionCallStrT = Callable[[ToolCall], str | None]
"""A callable to format a toolcall as a str"""

ApplyCallableT = (
    Callable[[PipelineMsgT], Any] | Callable[[PipelineMsgT, bool], Any] | Callable[[PipelineMsgT, bool, int], Any]
)
"""A function taking 1-3 args"""
