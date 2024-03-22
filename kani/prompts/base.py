import abc
from typing import Iterable

from kani.models import ChatRole
from kani.prompts.examples import PipelineExample, natural_join
from kani.prompts.types import PipelineMsgT, PredicateFilterT, RoleFilterT


class PipelineStep(abc.ABC):
    def execute(self, msgs: list[PipelineMsgT]):
        """Apply this step's effects on the pipeline."""
        raise NotImplementedError

    def explain(self) -> str:
        """Return a string explaining what this step does."""
        raise NotImplementedError

    def examples(self) -> list[PipelineExample]:
        """Return a list of examples the pipeline explain should include."""
        raise NotImplementedError

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
        if isinstance(self.role, ChatRole):
            if msg.role != self.role:
                return False
        elif self.role and msg.role not in self.role:
            return False

        # predicate
        if self.predicate is not None and not self.predicate(msg):
            return False

        # default
        return True

    # explain helper
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
