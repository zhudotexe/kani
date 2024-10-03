import functools
import itertools
import operator
import pprint
import time
from typing import Any, Callable, Generic, TypeVar, overload

from kani.ai_function import AIFunction
from kani.models import ChatMessage, ChatRole
from kani.prompts.base import PipelineStep
from kani.prompts.docutils import autoparams
from kani.prompts.examples import ALL_EXAMPLE_KWARGS, build_conversation, build_functions
from kani.prompts.steps import (
    Apply,
    ConversationDict,
    ConversationFmt,
    EnsureBoundFunctionCalls,
    EnsureStart,
    FunctionCallFmt,
    MacroApply,
    MergeConsecutive,
    Remove,
    TranslateRole,
    Wrap,
)
from kani.prompts.types import (
    ApplyCallableT,
    ApplyResultT,
    FunctionCallStrT,
    MacroApplyCallableT,
    MacroApplyResultT,
    MessageContentT,
    PipelineMsgT,
    PredicateFilterT,
    RoleFilterT,
)

# pre python3.11 Self type - let's just call it a PromptPipeline
try:
    from typing import Self
except ImportError:
    # noinspection PyTypeHints
    # this returns a forwardref to the class since it is unlikely that it will be subclassed, and it makes sphinx
    # sad to use a generic typevar here
    Self = "PromptPipeline"

# use a generic to specify the return type of the pipeline
T = TypeVar("T")


class PromptPipeline(Generic[T]):
    r"""
    This class creates a reproducible pipeline for translating a list of :class:`.ChatMessage` into an engine-specific
    format using fluent-style chaining.

    To build a pipeline, create an instance of ``PromptPipeline()`` and add steps by calling the step methods documented
    below. Most pipelines will end with a call to one of the terminals, which translates the intermediate form into
    the desired output format.

    **Usage**

    To use the pipeline, call the created pipeline object with a list of kani chat messages.

    To inspect the inputs/outputs of your pipeline, you can use :meth:`explain` to print a detailed explanation of the
    pipeline and multiple examples (selected based on the pipeline steps).

    **Example**

    Here's an example using the PromptPipeline to build a LLaMA 2 chat-style prompt:

    .. code-block:: python

        from kani import PromptPipeline, ChatRole

        pipe = (
            PromptPipeline()

            # System messages should be wrapped with this tag. We'll translate them to USER
            # messages since a system and user message go together in a single [INST] pair.
            .wrap(role=ChatRole.SYSTEM, prefix="<<SYS>>\n", suffix="\n<</SYS>>\n")
            .translate_role(role=ChatRole.SYSTEM, to=ChatRole.USER)

            # If we see two consecutive USER messages, merge them together into one with a
            # newline in between.
            .merge_consecutive(role=ChatRole.USER, sep="\n")
            # Similarly for ASSISTANT, but with a space (kani automatically strips whitespace from the ends of
            # generations).
            .merge_consecutive(role=ChatRole.ASSISTANT, sep=" ")

            # Finally, wrap USER and ASSISTANT messages in the instruction tokens. If our
            # message list ends with an ASSISTANT message, don't add the EOS token
            # (we want the model to continue the generation).
            .conversation_fmt(
                user_prefix="<s>[INST] ",
                user_suffix=" [/INST]",
                assistant_prefix=" ",
                assistant_suffix=" </s>",
                assistant_suffix_if_last="",
            )
        )

        # We can see what this pipeline does by calling explain()...
        pipe.explain()

        # And use it in our engine to build a string prompt for the LLM.
        prompt = pipe(ai.get_prompt())
    """

    def __init__(self, steps: list[PipelineStep] = None):
        if steps is None:
            steps = []
        self.steps = steps

    # ==== steps ====
    @overload
    def translate_role(
        self, *, to: ChatRole, warn: str = None, role: RoleFilterT = None, predicate: PredicateFilterT = None
    ) -> Self: ...

    @autoparams
    def translate_role(self, **kwargs):
        """
        Change the role of the matching messages.
        (e.g. for models which do not support native function calling, make all FUNCTION messages a USER message)

        :param to: The new role to translate the matching messages to.
        :param warn: A warning to emit if any messages are translated (e.g. if a model does not support certain roles).
        {ALL_FILTERS}
        """
        self.steps.append(TranslateRole(**kwargs))
        return self

    @overload
    def wrap(
        self, *, prefix: str = None, suffix: str = None, role: RoleFilterT = None, predicate: PredicateFilterT = None
    ) -> Self: ...

    @autoparams
    def wrap(self, **kwargs):
        """
        Wrap the matching messages with a given string prefix and/or suffix.

        For more fine-grained control over user/assistant message pairs as the last step in a pipeline, use
        :meth:`conversation_fmt` instead.

        :param prefix: The prefix to add before each matching message, if any.
        :param suffix: The suffix to add after each matching message, if any.
        {ALL_FILTERS}
        """
        self.steps.append(Wrap(**kwargs))
        return self

    @overload
    def merge_consecutive(
        self,
        *,
        sep: str = None,
        joiner: Callable[[list[PipelineMsgT]], MessageContentT] = None,
        out_role: ChatRole = None,
        role: RoleFilterT = None,
        predicate: PredicateFilterT = None,
    ) -> Self: ...

    @autoparams
    def merge_consecutive(self, **kwargs):
        r"""
        If multiple messages that match are found consecutively, merge them by either joining their contents with a
        string or call a joiner function.

        .. caution::
            If multiple roles are specified, this method will merge them as a group (e.g. if ``role=(USER, ASSISTANT)``,
            a USER message followed by an ASSISTANT message will be merged together into one with a role of
            ``out_role``).

            Similarly, if a predicate is specified, this method will merge all consecutive messages which match the
            given predicate.

        :param sep: The string to add between each matching message. Mutually exclusive with ``joiner``.
            If this is set, this is roughly equivalent to ``joiner=lambda msgs: sep.join(m.text for m in msgs)``.
        :param joiner: A function that will take a list of all messages in a consecutive group and return the final
            string. Mutually exclusive with ``sep``.
        :param out_role: The role of the merged message to use. This is required if multiple ``role``\ s are specified
            or ``role`` is not set; otherwise it defaults to the common role of the merged messages.
        {ALL_FILTERS}
        """
        self.steps.append(MergeConsecutive(**kwargs))
        return self

    @overload
    def function_call_fmt(
        self, func: FunctionCallStrT, *, prefix: str = "\n", sep: str = "", suffix: str = ""
    ) -> Self: ...

    def function_call_fmt(self, *args, **kwargs):
        """
        For each message with one or more requested tool calls, call the provided function on each requested tool call
        and append it to the message's content.

        :param func: A function taking a :class:`.ToolCall` and returning a string to append to the content of the
            message containing the requested call, or None to ignore the tool call.
        :param prefix: If at least one tool call is formatted, a prefix to insert after the message's contents and
            before the formatted string.
        :param sep: If two or more tool calls are formatted, the string to insert between them.
        :param suffix: If at least one tool call is formatted, a suffix to insert after the formatted string.
        """
        self.steps.append(FunctionCallFmt(*args, **kwargs))
        return self

    # removers
    @overload
    def remove(self, *, role: RoleFilterT = None, predicate: PredicateFilterT = None) -> Self: ...

    @autoparams
    def remove(self, **kwargs):
        """
        Remove all messages that match the filters from the output.

        {ALL_FILTERS}
        """
        self.steps.append(Remove(**kwargs))
        return self

    @overload
    def ensure_start(self, *, role: RoleFilterT = None, predicate: PredicateFilterT = None) -> Self: ...

    @autoparams
    def ensure_start(self, **kwargs):
        """
        Ensure that the output starts with a message with the given role by removing all messages from the start that
        do NOT match the given filters, such that the first message in the output matches.

        This should NOT be used to ensure that a system prompt is passed; the intent of this step is to prevent
        an orphaned FUNCTION result or ASSISTANT reply after earlier messages were context-managed out.

        {ALL_FILTERS}
        """
        self.steps.append(EnsureStart(**kwargs))
        return self

    @overload
    def ensure_bound_function_calls(self, id_translator: Callable[[str], str] = None) -> Self: ...

    @autoparams
    def ensure_bound_function_calls(self, *args, **kwargs):
        """
        Ensure that each FUNCTION message is preceded by an ASSISTANT message requesting it, and that each FUNCTION
        message's ``tool_call_id`` matches the request. If a FUNCTION message has no ``tool_call_id`` (e.g. a few-shot
        prompt), bind it to a preceding ASSISTANT message if it is unambiguous.

        Will remove hanging FUNCTION messages (i.e. messages where the corresponding request was managed out of the
        model's context) from the beginning of the prompt if necessary.

        :param id_translator: A function that takes a function ID (usually a UUID4 string) and returns a translated ID.
            Used for engines that require the function_call_id to be in particular formats (e.g., Mistral).
        :raises PromptError: if it is impossible to bind each function call to a request unambiguously.
        """
        self.steps.append(EnsureBoundFunctionCalls(*args, **kwargs))
        return self

    @overload
    def apply(
        self, func: ApplyCallableT, *, role: RoleFilterT = None, predicate: PredicateFilterT = None
    ) -> "PromptPipeline[list[ApplyResultT]]": ...

    @autoparams
    def apply(self, *args, **kwargs):
        """
        Apply the given function to all matched messages. Replace the message with the function's return value.

        The function may take 1-2 positional parameters: the first will always be the matched message at the current
        pipeline step, and the second will be the context this operation is occurring in (a :class:`.ApplyContext`).

        :param func: A function that takes 1-2 positional parameters ``(msg, ctx)`` that will be called
            on each matching message. If this function does not return a :class:`ChatMessage`, it should be the last
            step in the pipeline. If this function returns ``None``, the input message will be removed from the output.
        {ALL_FILTERS}
        """
        self.steps.append(Apply(*args, **kwargs))
        return self

    @overload
    def macro_apply(self, func: MacroApplyCallableT) -> "PromptPipeline[list[MacroApplyResultT]]": ...

    @autoparams
    def macro_apply(self, *args, **kwargs):
        """
        Apply the given function to the list of all messages in the pipeline.
        This step can effectively be used to create an ad-hoc step.

        The function must take 2 positional parameters: the first is the list of messages, and the second is
        the list of available functions.

        :param func: A function that takes 2 positional parameters ``(messages, functions)`` that will be called
            on the list of messages. If this function does not return a ``list[ChatMessage]``, it should be the
            last step in the pipeline.
        """
        self.steps.append(MacroApply(*args, **kwargs))
        return self

    # ==== terminals ====
    @overload
    def conversation_fmt(
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
    ) -> "PromptPipeline[str]": ...

    def conversation_fmt(self, **kwargs):
        """
        Takes in the list of messages and joins them into a single conversation-formatted string by:

        * wrapping messages with the defined prefixes/suffixes by role
        * joining the messages' contents with the defined *sep*
        * adding a generation suffix, if necessary.

        This method should be the last step in a pipeline and will cause the pipeline to return a :class:`str`.

        :param prefix: A string to insert once before the rest of the prompt, unconditionally.
        :param sep: A string to insert between messages, if any. Similar to ``sep.join(...)``.
        :param suffix: A string to insert once after the rest of the prompt, unconditionally.
        :param generation_suffix: A string to add to the end of the prompt to prompt the model to begin its turn.
        :param user_prefix: A prefix to add before each USER message.
        :param user_suffix: A suffix to add after each USER message.
        :param assistant_prefix: A prefix to add before each ASSISTANT message.
        :param assistant_suffix: A suffix to add after each ASSISTANT message.
        :param assistant_suffix_if_last: If not None and the prompt ends with an ASSISTANT message, this string will be
            added to the end of the prompt *instead* of the ``assistant_suffix + generation_suffix``.
            This is intended to allow consecutive ASSISTANT messages to continue generation from an unfinished prior
            message.
        :param system_prefix: A prefix to add before each SYSTEM message.
        :param system_suffix: A suffix to add after each SYSTEM message.
        :param function_prefix: A prefix to add before each FUNCTION message.
        :param function_suffix: A suffix to add after each FUNCTION message.
        """
        self.steps.append(ConversationFmt(**kwargs))
        return self

    @overload
    def conversation_dict(
        self,
        *,
        system_role: str = "system",
        user_role: str = "user",
        assistant_role: str = "assistant",
        function_role: str = "tool",
        content_transform: Callable[[ChatMessage], Any] = lambda msg: msg.text,
        additional_keys: Callable[[ChatMessage], dict] = lambda msg: {},
    ) -> "PromptPipeline[list[dict[str, Any]]]": ...

    def conversation_dict(self, **kwargs):
        """
        Takes in the list of messages and returns a list of dictionaries with ("role", "content") keys.

        By default, the "role" key will be "system", "user", "assistant", or "tool" unless the respective role
        override is specified.

        By default, the "content" key will be ``message.text`` unless the ``content_transform`` argument is specified.

        This method should be the last step in a pipeline and will cause the pipeline to return a ``list[dict]``.

        .. caution::
            By default, this step will truncate tool calling metadata! Use ``additional_keys`` to provide tool call
            requests on ASSISTANT messages and additional metadata like tool call IDs on FUNCTION messages.

        :param system_role: The role to give to SYSTEM messages (default "system").
        :param user_role: The role to give to USER messages (default "user").
        :param assistant_role: The role to give to ASSISTANT messages (default "assistant").
        :param function_role: The role to give to FUNCTION messages (default "tool").
        :param content_transform: A function taking in the message and returning the contents of the "content" key
            (defaults to ``msg.text``).
        :param additional_keys: A function taking in the message and returning a dictionary containing any additional
            keys to add to the message's dict.
        """
        self.steps.append(ConversationDict(**kwargs))
        return self

    # ==== eval ====
    def __call__(self, msgs: list[ChatMessage], functions: list[AIFunction] = None, **kwargs) -> T:
        """
        Apply the pipeline to a list of kani messages. The return type will vary based on the steps in the pipeline;
        if no steps are defined the return type will be a copy of the input messages.
        """
        return self.execute(msgs, functions, **kwargs)

    def execute(
        self, msgs: list[ChatMessage], functions: list[AIFunction] = None, *, deepcopy=False, for_measurement=False
    ) -> T:
        """
        Apply the pipeline to a list of kani messages. The return type will vary based on the steps in the pipeline;
        if no steps are defined the return type will be a copy of the input messages.

        This lower-level method offers more fine-grained control over the steps that are run (e.g. to measure the
        length of a single message).

        :param msgs: The messages to apply the pipeline to.
        :param functions: Any functions available to the model.
        :param deepcopy: Whether to deep-copy each message before running the pipeline.
        :param for_measurement: If the pipeline is being run to measure the length of a single message. In this case,
            any ``ensure_start`` steps will be ignored, and the returned message may not be a valid prompt - the only
            guarantee is on the length.
        """
        if functions is None:
            functions = []

        # let's use the lower-level model_copy() since we aren't changing anything
        data = [m.model_copy(deep=deepcopy) for m in msgs]

        # and apply the pipeline
        for step in self.steps:
            # for measurement: ignore ensure_start
            if for_measurement and isinstance(step, EnsureStart):
                continue

            # apply step
            data = step.execute(data, functions)

        # return the result
        return data

    # ==== utils ====
    def explain(
        self, example: list[ChatMessage] = None, functions: list[AIFunction] = None, *, all_cases=False, **kwargs
    ):
        """
        Print out a summary of the pipeline and an example conversation transformation based on the steps in the
        pipeline.

        .. caution::
            This method will run the pipeline on an example constructed based on the steps in this pipeline. You may
            encounter unexpected side effects if your pipeline uses :meth:`apply` with a function with side effects.
        """
        if functions is None:
            functions = []

        hdg = f"Prompt Pipeline ({len(self.steps)} steps)"
        print(f"{hdg}\n{'=' * len(hdg)}")
        listwidth = len(str(len(self.steps)))
        for idx, step in enumerate(self.steps):
            print(f"{idx + 1:>{listwidth}}. {step.explain()}")
        print()

        # build examples
        print("Example\n-------")
        if all_cases:
            example_kwargs = {k: True for k in ALL_EXAMPLE_KWARGS}
        else:
            example_kwargs = functools.reduce(operator.or_, (s.explain_example_kwargs() for s in self.steps), kwargs)

        if not example:
            examples_msg_grps = build_conversation(**example_kwargs)
            examples_msgs = list(itertools.chain.from_iterable(examples_msg_grps))
        else:
            examples_msg_grps = [example]
            examples_msgs = example

        if not functions:
            example_functions = build_functions(**example_kwargs)
        else:
            example_functions = functions

        # run and time example
        start = time.perf_counter()
        example_out = self(examples_msgs, example_functions)
        end = time.perf_counter()
        exec_time = end - start

        # print example i/o
        print(f"*Execution time: {exec_time * 1000:.3}ms*")
        print("### Input\n```py\n[", end="")
        for grp in examples_msg_grps:
            print()
            for msg in grp:
                print(f" {msg!r}")
        print("]\n```\n")

        print("### Output")
        if isinstance(example_out, str):
            print("```text")
            print(example_out)
        else:
            print("```py")
            pprint.pprint(example_out)
        print("```\n")

        # print note
        unused_kwargs = [k for k in ALL_EXAMPLE_KWARGS if k not in example_kwargs]
        if unused_kwargs and not example:
            print(
                "### Note\nSome edge cases are not represented in this example. To view all test cases, use"
                f" `.explain(all_cases=True)` or set any of the following keyword arguments: {unused_kwargs}.\n"
                "You may also specify your own test case with `.explain(example=...)`."
            )

    def __repr__(self):
        steps_fmt = pprint.pformat(self.steps)
        return f"{type(self).__name__}({steps_fmt})"

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, item):
        return self.steps[item]
