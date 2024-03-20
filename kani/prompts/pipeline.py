from kani.models import ChatMessage, ChatRole
from kani.prompts.docutils import autoparams
from kani.prompts.types import ApplyCallableT, FunctionCallStrT, PredicateFilterT, RoleFilterT


class PromptPipeline:
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

    **Automatic Optimization**

    The first time the pipeline is applied, it will automatically compile the intermediate operations to optimize
    the pipeline (e.g. reducing redundant loops and merging nonconflicting operations). To compile this eagerly, call
    :meth:`compile` as the last step in your pipeline.

    **Example**

    Here's an example using the PromptPipeline to build a LLaMA 2 chat-style prompt:

    .. code-block:: python

        pipe = (
            PromptPipeline()

            # System messages should be wrapped with this tag. We'll translate them to USER
            # messages since a system and user message go together in a single [INST] pair.
            .wrap(role=ChatRole.SYSTEM, prefix="<<SYS>>\n", suffix="\n<</SYS>>\n")
            .translate_role(role=ChatRole.SYSTEM, to=ChatRole.USER)

            # If we see two consecutive USER messages, merge them together into one with a
            # newline in between.
            .merge_consecutive(role=ChatRole.USER, sep="\n")
            # Similarly for ASSISTANT, but simply concatenate them.
            .merge_consecutive(role=ChatRole.ASSISTANT, sep="")

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

    def __init__(self):
        self.steps = []

    # ==== steps ====
    @autoparams
    def translate_role(self, *, to: ChatRole, role: RoleFilterT = None, predicate: PredicateFilterT = None):
        """
        Change the role of the matching messages.
        (e.g. for models which do not support native function calling, make all FUNCTION messages a USER message)

        :param to: The new role to translate the matching messages to.
        {ALL_FILTERS}
        """
        # self.steps.append(TranslateRole(from_, to))
        return self

    @autoparams
    def wrap(
        self, *, prefix: str = None, suffix: str = None, role: RoleFilterT = None, predicate: PredicateFilterT = None
    ):
        """
        Wrap the matching messages with a given string prefix and/or suffix.

        For more fine-grained control over user/assistant message pairs as the last step in a pipeline, use
        :meth:`conversation_fmt` instead.

        :param prefix: The prefix to add before each matching message, if any.
        :param suffix: The suffix to add after each matching message, if any.
        {ALL_FILTERS}
        """
        return self

    @autoparams
    def merge_consecutive(
        self, *, sep: str, out_role: ChatRole = None, role: RoleFilterT = None, predicate: PredicateFilterT = None
    ):
        r"""
        If multiple messages that match are found consecutively, merge them into a single message whose contents
        are equivalent to the contents of the merged messages joined by ``sep``.

        .. note::
            This method will not merge FUNCTION messages as they contain metadata that cannot be merged with other
            messages. If you need to merge FUNCTION messages, you should translate them into another role first.

        .. caution::
            If multiple roles are specified, this method will merge them as a group (e.g. if ``role=(USER, ASSISTANT)``,
            a USER message followed by an ASSISTANT message will be merged together into one with a role of
            ``out_role``).

            Similarly, if a predicate is specified, this method will merge all consecutive messages which match the
            given predicate.

        :param sep: The prefix to add before each matching message, if any.
        :param out_role: The role of the merged message to use. This is required if multiple ``role``\ s are specified
            or ``role`` is not set; otherwise it defaults to the common role of the merged messages.
        {ALL_FILTERS}
        """
        return self

    def function_call_fmt(self, func: FunctionCallStrT, *, prefix: str = "\n", sep: str = "", suffix: str = None):
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
        return self

    # removers
    @autoparams
    def remove(self, *, role: RoleFilterT = None, predicate: PredicateFilterT = None):
        """
        Remove all messages that match the filters from the output.

        {ALL_FILTERS}
        """
        return self

    @autoparams
    def ensure_start(self, *, role: RoleFilterT = None, predicate: PredicateFilterT = None):
        """
        Ensure that the output starts with a message with the given role by removing all messages from the start that
        do NOT match the given filters, such that the first message in the output matches.

        This should NOT be used to ensure that a system prompt is passed; the intent of this step is to prevent
        an orphaned FUNCTION result or ASSISTANT reply after earlier messages were context-managed out.

        {ALL_FILTERS}
        """
        return self

    @autoparams
    def apply(self, func: ApplyCallableT, *, role: RoleFilterT = None, predicate: PredicateFilterT = None):
        """
        Apply the given function to all matched messages.

        The function may take 1-3 positional parameters: the first will always be the matched message at the current
        pipeline step, the second will be whether or not the message is the last one in the list of messages, and the
        third will be the index of the message in the list of messages.

        :param func: A function that takes 1-3 positional parameters ``(msg, is_last, idx)`` that will be called
            on each matching message. If this function does not return a :class:`ChatMessage`, it should be the last
            step in the pipeline. If this function returns ``None``, the input message will be removed from the output.
        {ALL_FILTERS}
        """
        return self

    # ==== terminals ====
    def conversation_fmt(
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
        """
        Takes in the list of messages and joins them into a single conversation-formatted string by:

        * wrapping messages with the defined prefixes/suffixes by role
        * joining the messages' contents with the defined *sep*
        * adding a generation suffix, if necessary.

        This method should be the last step in a pipeline and will cause the pipeline to return a :class:`str`.

        :param sep: A string to insert between messages, if any. Similar to ``sep.join(...)``.
        :param generation_suffix: A string to add to the end of the prompt to prompt the model to begin its turn.
        :param user_prefix: A prefix to add before each USER message.
        :param user_suffix: A suffix to add after each USER message.
        :param assistant_prefix: A prefix to add before each ASSISTANT message.
        :param assistant_suffix: A suffix to add after each ASSISTANT message.
        :param assistant_suffix_if_last: If not None and the prompt ends with an ASSISTANT message, this string will be
            added to the end of the prompt *instead* of the ``generation_suffix``. This is intended to allow consecutive
            ASSISTANT messages to continue generation from an unfinished prior message.
        :param system_prefix: A prefix to add before each SYSTEM message.
        :param system_suffix: A suffix to add after each SYSTEM message.
        :param function_prefix: A prefix to add before each USER message.
        :param function_suffix: A suffix to add after each USER message.
        """
        return self

    # ==== eval ====
    def __call__(self, msgs: list[ChatMessage]):
        """
        Apply the pipeline to a list of kani messages. The return type will vary based on the steps in the pipeline;
        if no steps are defined the return type will be a copy of the input messages.
        """
        # let's use the lower-level model_copy() to preserve IDs since these are temporary mutable messages
        data = [m.model_copy() for m in msgs]

        # and apply the pipeline
        # TODO compilation
        for step in self.steps:
            data = step.execute(data)

        # return the result
        return data

    # ==== utils ====
    def explain(self):
        """
        Print out a summary of the pipeline and test cases based on the steps in the pipeline.

        TODO: return a list of test case summaries for unit testing?
        """
        # TODO
