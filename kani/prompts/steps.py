import inspect
import itertools
import warnings
from typing import Any, Callable

from kani.ai_function import AIFunction
from kani.exceptions import PromptError
from kani.models import ChatMessage, ChatRole
from kani.prompts.base import FilterMixin, PipelineStep
from kani.prompts.types import (
    ApplyCallableT,
    ApplyContext,
    ApplyResultT,
    FunctionCallStrT,
    MacroApplyCallableT,
    MessageContentT,
    PipelineMsgT,
)


# ==== steps ====
class TranslateRole(FilterMixin, PipelineStep):
    def __init__(self, *, to: ChatRole, warn: str = None, **filter_kwargs):
        super().__init__(**filter_kwargs)
        self.to = to
        self.warn = warn

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[PipelineMsgT]:
        has_warned = False
        for msg in self.filtered(msgs):
            if self.warn and not has_warned:
                has_warned = True
                warnings.warn(self.warn)
            msg.role = self.to
        return msgs

    def explain(self) -> str:
        return f"Translate all {self.explain_note()} to {self.to.value} messages"


class Wrap(FilterMixin, PipelineStep):
    def __init__(self, *, prefix: str = None, suffix: str = None, **filter_kwargs):
        super().__init__(**filter_kwargs)
        self.prefix = prefix
        self.suffix = suffix

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[PipelineMsgT]:
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
    def __init__(
        self,
        *,
        sep: str = None,
        joiner: Callable[[list[PipelineMsgT]], MessageContentT] = None,
        out_role: ChatRole = None,
        **filter_kwargs,
    ):
        super().__init__(**filter_kwargs)
        if sep is not None and joiner is not None:
            raise ValueError("Only one of (sep, joiner) may be set.")
        if sep is None and joiner is None:
            raise ValueError("You must set at least one of (sep, joiner).")

        self.sep = sep
        self.joiner = joiner
        self.out_role = out_role

        if self.out_role is None:
            if isinstance(self.role, ChatRole):
                self.out_role = self.role
            else:
                raise ValueError(
                    "You must supply an `out_role` if a `.merge_consecutive` pipeline step can match messages with"
                    " different roles."
                )

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[PipelineMsgT]:
        out = []

        # group messages by whether they match the filter, putting consecutive matching messages into a list
        for matching, group_msgs in itertools.groupby(msgs, key=self.matches_filter):
            group_msgs = list(group_msgs)
            # nonmatching messages get sent to output unchanged
            if not matching:
                out.extend(group_msgs)
            # >1 consecutive matching messages get merged
            else:
                out.append(self._do_merge(group_msgs))

        return out

    def explain(self) -> str:
        if self.sep:
            how = f"by inserting {self.sep!r} between them"
        else:
            how = "by calling the given function"

        if self.out_role:
            return f"Merge consecutive {self.explain_note()} into a single {self.out_role.value} message {how}"
        return f"Merge consecutive {self.explain_note()} into a single message {how}"

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

    # helper
    def _do_merge(self, msgs: list[PipelineMsgT]) -> PipelineMsgT:
        """Helper to merge multiple messages' content, respecting if it has parts or not"""
        if len(msgs) < 1:
            raise ValueError("At least one message must be supplied to merge")

        # if we're doing sep and only have 1 message, just ignore it and return it unchanged
        if self.sep and len(msgs) == 1:
            return msgs[0]

        has_parts = any(isinstance(m.content, list) for m in msgs)

        # we'll use the first message as the returned one
        out_msg = msgs[0]
        out_msg.role = self.out_role
        out_msg.tool_call_id = None  # if we are merging function calls, something is wacky and we should strip this
        if not all(m.tool_calls is None for m in msgs):
            out_msg.tool_calls = list(itertools.chain.from_iterable(m.tool_calls or [] for m in msgs))

        # SEP
        if self.sep:
            # all text: replace the first message and return it
            if not has_parts:
                out_msg.content = self.sep.join(m.content or "" for m in msgs)
                return out_msg

            # otherwise, build up a list, inserting sep between each
            out_msg.content = out_msg.parts
            for msg in msgs[1:]:
                out_msg.content.append(self.sep)
                out_msg.content.extend(msg.parts)
        # JOINER
        else:
            out_msg.content = self.joiner(msgs)
        return out_msg


class FunctionCallFmt(PipelineStep):
    def __init__(self, func: FunctionCallStrT, *, prefix: str = "\n", sep: str = "", suffix: str = ""):
        self.func = func
        self.prefix = prefix
        self.sep = sep
        self.suffix = suffix

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[PipelineMsgT]:
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

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[PipelineMsgT]:
        return [m for m in msgs if not self.matches_filter(m)]

    def explain(self) -> str:
        return f"Remove all {self.explain_note()}"


class EnsureStart(FilterMixin, PipelineStep):
    def __init__(self, **filter_kwargs):
        super().__init__(**filter_kwargs)

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[PipelineMsgT]:
        first_matching_idx = next((i for i, msg in enumerate(msgs) if self.matches_filter(msg)), len(msgs))
        return msgs[first_matching_idx:]

    def explain(self) -> str:
        return f"Ensure that the prompt starts with a {self.explain_note('or', plural=False)}"


class EnsureBoundFunctionCalls(PipelineStep):
    def __init__(self, id_translator: Callable[[str], str] = None):
        super().__init__()
        self.id_translator = id_translator

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[PipelineMsgT]:
        free_toolcall_ids = set()
        out = []
        for m in msgs:
            # if this is not a function result and there are free tool call IDs, raise
            if m.role != ChatRole.FUNCTION and free_toolcall_ids:
                raise PromptError(
                    f"Encountered a {m.role.value!r} message but expected a FUNCTION message to satisfy the pending"
                    f" tool call(s): {free_toolcall_ids}"
                )
            # asst: add tool call IDs to freevars
            if m.role == ChatRole.ASSISTANT and m.tool_calls:
                for tc in m.tool_calls:
                    free_toolcall_ids.add(tc.id)
            # func: bind freevars
            elif m.role == ChatRole.FUNCTION:
                # has ID: bind it if requested; yeet if not
                if m.tool_call_id is not None:
                    if m.tool_call_id in free_toolcall_ids:
                        free_toolcall_ids.remove(m.tool_call_id)
                    else:
                        # this happens if the tool call is pushed out of context but the result is still here,
                        # and we have always included messages beforehand
                        continue  # yeet this message
                # no ID: bind if unambiguous
                elif len(free_toolcall_ids) == 1:
                    m.tool_call_id = free_toolcall_ids.pop()
                # no ID: error if ambiguous
                elif len(free_toolcall_ids) > 1:
                    raise PromptError(
                        "Got a FUNCTION message with no tool_call_id but multiple tool calls are pending"
                        f" ({free_toolcall_ids})! Set the tool_call_id to resolve the pending tool requests."
                    )
                # otherwise pass the FUNCTION message through

            # translate the id(s) if needed
            if self.id_translator is not None:
                if m.tool_calls is not None:
                    # copy the tool calls here to prevent unintended mutation
                    m.tool_calls = [tc.copy_with(id=self.id_translator(tc.id)) for tc in m.tool_calls]
                if m.tool_call_id is not None:
                    m.tool_call_id = self.id_translator(m.tool_call_id)

            out.append(m)
        return out

    def explain(self) -> str:
        return "Ensure that each function call is bound"

    def explain_example_kwargs(self) -> dict[str, bool]:
        return {"function_call": True}


class Apply(FilterMixin, PipelineStep):
    def __init__(self, func: ApplyCallableT, **filter_kwargs):
        super().__init__(**filter_kwargs)
        self.func = func

        # func introspection: generate a wrapper for the right number of args provided
        sig = inspect.signature(func)
        if len(sig.parameters) == 1:
            self.func_wrapped = lambda msg, ctx: self.func(msg)
        elif len(sig.parameters) == 2:
            self.func_wrapped = self.func
        else:
            raise ValueError(
                "The applied function must have 1 to 2 positional parameters (msg, ctx) (got a function with"
                f" {len(sig.parameters)} parameters)."
            )

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[ApplyResultT]:
        out = []
        for i, msg in enumerate(msgs):
            # for each matching message, append f(msg) if it's not None
            if self.matches_filter(msg):
                ctx = ApplyContext(msg=msg, is_last=i == len(msgs) - 1, idx=i, messages=msgs, functions=functions)
                replacement = self.func_wrapped(msg, ctx)
                if replacement is not None:
                    out.append(replacement)
            # else just append the msg unchanged
            else:
                out.append(msg)
        return out

    def explain(self) -> str:
        return f"Apply the given function to each {self.explain_note('and', plural=False)}"


class MacroApply(PipelineStep):
    def __init__(self, func: MacroApplyCallableT):
        self.func = func

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[ApplyResultT]:
        return self.func(msgs, functions)

    def explain(self) -> str:
        return "Apply the given function to the list of all messages in the pipeline"


# ==== terminals ====
class ConversationFmt(PipelineStep):
    def __init__(
        self,
        *,
        # general formatting
        prefix: str = "",
        sep: str = "",
        suffix: str = "",
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
        self.prefix = prefix
        self.sep = sep
        self.suffix = suffix
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

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> str:
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
        return self.prefix + self.sep.join(parts) + self.suffix

    def explain(self) -> str:
        return "Format the messages into a single conversation-formatted string (see example)"

    def explain_example_kwargs(self) -> dict[str, bool]:
        kwargs = {}
        if self.function_prefix != self.user_prefix or self.function_suffix != self.user_suffix:
            kwargs["function_call"] = True
        return kwargs


class ConversationDict(PipelineStep):
    def __init__(
        self,
        *,
        system_role: str = "system",
        user_role: str = "user",
        assistant_role: str = "assistant",
        function_role: str = "tool",
        content_transform: Callable[[ChatMessage], Any] = lambda msg: msg.text,
        additional_keys: Callable[[ChatMessage], dict] = lambda msg: {},
    ):
        self.role_to_str = {
            ChatRole.SYSTEM: system_role,
            ChatRole.USER: user_role,
            ChatRole.ASSISTANT: assistant_role,
            ChatRole.FUNCTION: function_role,
        }
        self.content_transform = content_transform
        self.additional_keys = additional_keys

    def execute(self, msgs: list[PipelineMsgT], functions: list[AIFunction]) -> list[dict[str, Any]]:
        out = []
        for msg in msgs:
            msg_dict = {
                "role": self.role_to_str[msg.role],
                "content": self.content_transform(msg),
                **self.additional_keys(msg),
            }
            out.append(msg_dict)
        return out

    def explain(self) -> str:
        return 'Return the messages as dictionaries with {"role": ..., "content": ...} keys (see example)'

    def explain_example_kwargs(self) -> dict[str, bool]:
        kwargs = {}
        if self.role_to_str[ChatRole.FUNCTION] != "function":
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
