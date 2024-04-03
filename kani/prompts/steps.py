import inspect
import itertools

from kani.models import ChatRole
from kani.prompts.base import FilterMixin, PipelineStep
from kani.prompts.types import ApplyCallableT, FunctionCallStrT, PipelineMsgT


# ==== steps ====
class TranslateRole(FilterMixin, PipelineStep):
    def __init__(self, *, to: ChatRole, **filter_kwargs):
        super().__init__(**filter_kwargs)
        self.to = to

    def execute(self, msgs: list[PipelineMsgT]) -> list[PipelineMsgT]:
        for msg in self.filtered(msgs):
            msg.role = self.to
        return msgs

    def explain(self) -> str:
        return f"Translate all {self.explain_note()} to {self.to.value} messages"


class Wrap(FilterMixin, PipelineStep):
    def __init__(self, *, prefix: str = None, suffix: str = None, **filter_kwargs):
        super().__init__(**filter_kwargs)
        self.prefix = prefix
        self.suffix = suffix

    def execute(self, msgs: list[PipelineMsgT]) -> list[PipelineMsgT]:
        for msg in self.filtered(msgs):
            _wrap_content_inplace(msg, self.prefix, self.suffix)
        return msgs

    def explain(self) -> str:
        if self.prefix and self.suffix:
            return f"Wrap each {self.explain_note('or', plural=False)} with {self.prefix!r} and {self.suffix!r}"
        elif self.prefix:
            return f"Add {self.prefix!r} to the start of each {self.explain_note('or', plural=False)}"
        elif self.suffix:
            return f"Add {self.suffix!r} to the end of each {self.explain_note('or', plural=False)}"
        return "This step doesn't do anything as configured - you should supply a prefix or suffix."


class MergeConsecutive(FilterMixin, PipelineStep):
    def __init__(self, *, sep: str, out_role: ChatRole = None, **filter_kwargs):
        super().__init__(**filter_kwargs)
        self.sep = sep
        self.out_role = out_role

        if self.out_role is None:
            if isinstance(self.role, ChatRole):
                self.out_role = self.role
            else:
                raise ValueError(
                    "You must supply an `out_role` if a `.merge_consecutive` pipeline step can match messages with"
                    " different roles."
                )

    def execute(self, msgs: list[PipelineMsgT]) -> list[PipelineMsgT]:
        out = []

        # group messages by whether they match the filter, putting consecutive matching messages into a list
        for matching, group_msgs in itertools.groupby(msgs, key=self.matches_filter):
            group_msgs = list(group_msgs)
            # nonmatching messages get sent to output unchanged
            if not matching:
                out.extend(group_msgs)
            # only 1 consecutive matching message, send to output
            elif len(group_msgs) == 1:
                out.extend(group_msgs)
            # >1 consecutive matching messages get merged
            else:
                out.append(_merge_messages(group_msgs, self.sep, self.out_role))

        return out

    def explain(self) -> str:
        if self.out_role:
            return (
                f"Merge consecutive {self.explain_note()} into a single {self.out_role.value} message by inserting"
                f" {self.sep!r} between them"
            )
        return f"Merge consecutive {self.explain_note()} into a single message by inserting {self.sep!r} between them"

    def explain_example_kwargs(self) -> dict[str, bool]:
        kwargs = super().explain_example_kwargs()
        if self.matches_role(ChatRole.USER):
            kwargs["consecutive_user"] = True
        if self.matches_role(ChatRole.ASSISTANT):
            kwargs["consecutive_assistant"] = True
        if self.matches_role(ChatRole.SYSTEM):
            kwargs["consecutive_system"] = True
        if self.matches_role(ChatRole.FUNCTION):
            kwargs["multi_function_call"] = True
        return kwargs


class FunctionCallFmt(PipelineStep):
    def __init__(self, func: FunctionCallStrT, *, prefix: str = "\n", sep: str = "", suffix: str = ""):
        self.func = func
        self.prefix = prefix
        self.sep = sep
        self.suffix = suffix

    def execute(self, msgs: list[PipelineMsgT]) -> list[PipelineMsgT]:
        # for each message with 1+ tool calls,
        for msg in msgs:
            if not msg.tool_calls:
                continue

            # call the func on each tc
            tc_fmt = (fmt for tc in msg.tool_calls if (fmt := self.func(tc)) is not None)
            # join and append to the message
            tc_content = self.sep.join(tc_fmt)
            _wrap_content_inplace(msg, prefix=None, suffix=f"{self.prefix}{tc_content}{self.suffix}")

        return msgs

    def explain(self) -> str:
        fmt_repr = f"{self.prefix}{{{self.sep!r}.join(f(tc) for tc in msg.tool_calls)}}{self.suffix}"
        return f"Apply the given function to each tool call, and append {fmt_repr!r} to each message with tool calls"

    def explain_example_kwargs(self) -> dict[str, bool]:
        kwargs = {"function_call": True}
        if self.sep:
            kwargs["multi_function_call"] = True
        return kwargs


class Remove(FilterMixin, PipelineStep):
    def __init__(self, **filter_kwargs):
        super().__init__(**filter_kwargs)

    def execute(self, msgs: list[PipelineMsgT]) -> list[PipelineMsgT]:
        return [m for m in msgs if not self.matches_filter(m)]

    def explain(self) -> str:
        return f"Remove all {self.explain_note()}"


class EnsureStart(FilterMixin, PipelineStep):
    def __init__(self, **filter_kwargs):
        super().__init__(**filter_kwargs)

    def execute(self, msgs: list[PipelineMsgT]) -> list[PipelineMsgT]:
        first_matching_idx = next((i for i, msg in enumerate(msgs) if self.matches_filter(msg)), len(msgs))
        return msgs[first_matching_idx:]

    def explain(self) -> str:
        return f"Ensure that the prompt starts with a {self.explain_note('or', plural=False)}"


class Apply(FilterMixin, PipelineStep):
    def __init__(self, func: ApplyCallableT, **filter_kwargs):
        super().__init__(**filter_kwargs)
        self.func = func

        # func introspection: generate a wrapper for the right number of args provided
        sig = inspect.signature(func)
        if len(sig.parameters) == 1:
            self.func_wrapped = lambda msg, is_last, idx: self.func(msg)
        elif len(sig.parameters) == 2:
            self.func_wrapped = lambda msg, is_last, idx: self.func(msg, is_last)
        elif len(sig.parameters) == 3:
            self.func_wrapped = self.func
        else:
            raise ValueError(
                "The applied function must have 1 to 3 positional parameters (msg, is_last, idx) (got a function with"
                f" {len(sig.parameters)} parameters)."
            )

    def execute(self, msgs: list[PipelineMsgT]) -> list[PipelineMsgT]:
        out = []
        for i, msg in enumerate(msgs):
            # for each matching message, append f(msg) if it's not None
            if self.matches_filter(msg):
                replacement = self.func_wrapped(msg, i == len(msgs) - 1, i)
                if replacement is not None:
                    out.append(replacement)
            # else just append the msg unchanged
            else:
                out.append(msg)
        return out

    def explain(self) -> str:
        return f"Apply the given function to each {self.explain_note('and', plural=False)}"


# ==== terminals ====
class ConversationFmt(PipelineStep):
    def __init__(
        self,
        *,
        # general formatting
        sep: str = "",
        generation_suffix: str = "",
        # message-specific formatting
        # USER messages
        user_prefix: str = "",
        user_suffix: str = "",
        # ASSISTANT messages
        assistant_prefix: str = "",
        assistant_suffix: str = "",
        assistant_suffix_if_last: str = None,
        # SYSTEM messages
        system_prefix: str = "",
        system_suffix: str = "",
        # FUNCTION messages (if not specified, defaults to user)
        function_prefix: str = None,
        function_suffix: str = None,
    ):
        self.sep = sep
        self.generation_suffix = generation_suffix
        self.user_prefix = user_prefix
        self.user_suffix = user_suffix
        self.assistant_prefix = assistant_prefix
        self.assistant_suffix = assistant_suffix
        self.assistant_suffix_if_last = assistant_suffix_if_last
        self.system_prefix = system_prefix
        self.system_suffix = system_suffix
        self.function_prefix = function_prefix if function_prefix is not None else user_prefix
        self.function_suffix = function_suffix if function_suffix is not None else user_suffix

    def execute(self, msgs: list[PipelineMsgT]) -> str:
        parts = []
        for idx, msg in enumerate(msgs):
            if msg.role == ChatRole.USER:
                parts.append(f"{self.user_prefix}{msg.text}{self.user_suffix}")
            elif msg.role == ChatRole.ASSISTANT:
                # if this is the last message and we want custom ends, use the last suffix instead
                if idx == len(msgs) - 1 and self.assistant_suffix_if_last is not None:
                    assistant_suffix = self.assistant_suffix_if_last
                else:
                    assistant_suffix = self.assistant_suffix
                parts.append(f"{self.assistant_prefix}{msg.text}{assistant_suffix}")
            elif msg.role == ChatRole.SYSTEM:
                parts.append(f"{self.system_prefix}{msg.text}{self.system_suffix}")
            else:  # function
                parts.append(f"{self.function_prefix}{msg.text}{self.function_suffix}")

        # generation suffix if we aren't ending on an assistant message with a special asst_suffix_if_last
        if not (msgs and msgs[-1].role == ChatRole.ASSISTANT and self.assistant_suffix_if_last is not None):
            parts.append(self.generation_suffix)

        # join
        return self.sep.join(parts)

    def explain(self) -> str:
        return "Format the messages into a single conversation-formatted string (see example)"

    def explain_example_kwargs(self) -> dict[str, bool]:
        kwargs = {}
        if self.function_prefix != self.user_prefix or self.function_suffix != self.user_suffix:
            kwargs["function_call"] = True
        return kwargs


# ==== helpers ====
def _wrap_content_inplace(msg: PipelineMsgT, prefix: str | None, suffix: str | None):
    """Helper to add a prefix/suffix to a message's contents, respecting if it has parts or not"""
    if isinstance(msg.content, str) or msg.content is None:
        msg.content = f"{prefix or ''}{msg.content or ''}{suffix or ''}"
    # list
    else:
        if prefix:
            msg.content.insert(0, prefix)
        if suffix:
            msg.content.append(suffix)


def _merge_messages(msgs: list[PipelineMsgT], sep: str, out_role: ChatRole) -> PipelineMsgT:
    """Helper to merge multiple messages' content, respecting if it has parts or not"""
    if len(msgs) < 1:
        raise ValueError("At least one message must be supplied to merge")

    has_parts = any(isinstance(m.content, list) for m in msgs)

    # we'll use the first message as the returned one
    out_msg = msgs[0]
    out_msg.role = out_role
    out_msg.tool_call_id = None  # if we are merging function calls, something is wacky and we should strip this
    if not all(m.tool_calls is None for m in msgs):
        out_msg.tool_calls = list(itertools.chain.from_iterable(m.tool_calls or [] for m in msgs))

    # all text: replace the first message and return it
    if not has_parts:
        out_msg.content = sep.join(m.content or "" for m in msgs)
        return out_msg

    # otherwise, build up a list, inserting sep between each
    out_msg.content = out_msg.parts
    for msg in msgs[1:]:
        out_msg.content.append(sep)
        out_msg.content.extend(msg.parts)
    return out_msg
