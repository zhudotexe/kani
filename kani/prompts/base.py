import abc
from typing import Iterable

from kani.ai_function import AIFunction
from kani.models import ChatRole
from kani.prompts.types import PipelineMsgT, PredicateFilterT, RoleFilterT


class PipelineStep(abc.ABC):
    """
    The base class for all pipeline steps.

    If needed, you can subclass this and manually add steps to a :class:`.PromptPipeline`, but this is generally not
    necessary (consider using :meth:`.PromptPipeline.apply` instead).
    """

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]):
        """Apply this step's effects on the pipeline."""
        raise NotImplementedError

    def explain(self) -> str:
        """Return a string explaining what this step does."""
        raise NotImplementedError

    def explain_example_kwargs(self) -> dict[str, bool]:
        """Return a dict of kwargs to pass to examples.build_conversation to ensure relevant examples are included."""
        return {}

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({attrs})"


class FilterMixin:
    """helper mixin to implement filtering operations"""

    def __init__(self, role: RoleFilterT = None, predicate: PredicateFilterT = None):
        self.role = role
        self.predicate = predicate

    def filtered(self, msgs: list[PipelineMsgT]) -> Iterable[PipelineMsgT]:
        """Yield all messages that match the filter"""
        for msg in msgs:
            if self.matches_filter(msg):
                yield msg

    def matches_filter(self, msg: PipelineMsgT) -> bool:
        """Whether or not a message matches the filter"""
        # role(s)
        if not self.matches_role(msg.role):
            return False

        # predicate
        if self.predicate is not None and not self.predicate(msg):
            return False

        # default
        return True

    def matches_role(self, role: ChatRole) -> bool:
        """Whether or not this filter unconditionally matches the given role (only checks *role*, not *predicate*)"""
        if isinstance(self.role, ChatRole):
            return role == self.role
        elif self.role:
            return role in self.role
        return True

    # explain helpers
    def explain_note(self, join_sep="or", plural=True) -> str:
        """Returns a short note with the conditions this step applies to.
        (e.g. "messages" or "system messages that match the given predicate")

        :param join_sep: If the filter applies to more than one role, what word to use to join the role names
        :param plural: e.g. "each message" vs "all messages"
        """
        out = "messages" if plural else "message"

        # role(s)
        if isinstance(self.role, ChatRole):
            out = f"{self.role.value} {out}"
        elif self.role:
            msgs = natural_join([r.value for r in self.role], join_sep)
            out = f"{msgs} {out}"

        # predicate
        if self.predicate is not None:
            out += f" that {'match' if plural else 'matches'} the given predicate"

        return out

    # by default, let's include function call if any filtered step targets functions
    def explain_example_kwargs(self) -> dict[str, bool]:
        if self.matches_role(ChatRole.FUNCTION):
            return {"function_call": True}
        return {}


def natural_join(elems: list[str], sep: str):
    sep = f" {sep} "
    if len(elems) < 3:
        return sep.join(elems)
    return ", ".join(elems[:-1]) + f",{sep}{elems[-1]}"
