import logging
import re
import warnings

from kani import ReasoningPart
from kani.engines import Completion, WrapperEngine
from kani.engines.base import BaseCompletion
from kani.models import ChatMessage, MessagePart, ToolCall

log = logging.getLogger(__name__)


class BaseParser(WrapperEngine):
    """
    Abstract base class for model-specific tool call/reasoning parsers.

    To implement your own tool call/reasoning parser, subclass this class and:

    * implement ``parse_tool_calls(self, content: str) -> tuple[str, list[ToolCall]]``
    * implement ``parse_reasoning(self, content: str) -> list[MessagePart]``
    * pass default values of ``tool_call_start_token``, ``tool_call_end_token``, ``reasoning_start_token``, and
      ``reasoning_end_token`` to ``super().__init__(...)``

    This class will handle calling the parser and interrupting streams when tool calls/reasoning are detected.
    """

    def __init__(
        self,
        *args,
        tool_call_start_token: str | None = None,
        tool_call_end_token: str | None = None,
        reasoning_start_token: str | None = None,
        reasoning_end_token: str | None = None,
        reasoning_always_at_start=False,
        show_reasoning_in_stream=False,
        reasoning_in_stream_color=True,
        **kwargs,
    ):
        """
        :param tool_call_start_token: The token used to delimit the start of a tool call.
            Used to determine when to buffer streams.
        :param tool_call_end_token: The token used to delimit the end of a tool call.
            Used to determine when to yield streams.
        :param reasoning_start_token: The token used to delimit the start of a reasoning segment.
            Used to determine when to buffer streams.
        :param reasoning_end_token: The token used to delimit the end of a reasoning segment.
            Used to determine when to buffer streams and implement default reasoning parsing behaviour.
        :param reasoning_always_at_start: Whether the model's response always starts with reasoning and should be
            buffered until the ``reasoning_end_token`` is seen while streaming.
        :param show_reasoning_in_stream: Whether reasoning tokens should be yielded during streams. By default, only
            non-reasoning tokens will be yielded, and reasoning tokens will be included in a :class:`.ReasoningPart` in
            the final :class:`.ChatMessage`. This does not change the final returned :class:`.ChatMessage`, it only
            affects streamed tokens.
        :param reasoning_in_stream_color: If True, wraps yielded reasoning tokens in an ANSI color code to make them
            appear gray when printed in a terminal. Only takes effect when ``show_reasoning_in_stream=True``.
        """
        super().__init__(*args, **kwargs)
        self.tool_call_start_token = tool_call_start_token
        self.tool_call_end_token = tool_call_end_token
        self.reasoning_start_token = reasoning_start_token
        self.reasoning_end_token = reasoning_end_token
        self.reasoning_always_at_start = reasoning_always_at_start
        self.show_reasoning_in_stream = show_reasoning_in_stream
        self.reasoning_in_stream_color = reasoning_in_stream_color

        # build the regex for the splitter
        splitter_re_parts = []
        for attr in ("tool_call_start_token", "tool_call_end_token", "reasoning_start_token", "reasoning_end_token"):
            if (val := getattr(self, attr)) is not None:
                splitter_re_parts.append(re.escape(val))
        splitter_re_parts = "|".join(splitter_re_parts)
        self.splitter_re = re.compile(rf"({splitter_re_parts})") if splitter_re_parts else None

        # a moderate hack; globally save that we have initialized some parser
        from kani import model_specific

        model_specific._has_initialized_model_specific_parser = True

    def parse_tool_calls(self, content: str) -> tuple[str, list[ToolCall]]:
        """Given the string completion of the model, return the content without tool calls and the parsed tool calls."""
        # by default, do nothing
        return content, []

    def parse_reasoning(self, content: str) -> str | list[MessagePart | str]:
        """
        Given the string completion of the model (after parsing tool calls), return the content with reasoning
        transformed to ReasoningParts.
        """
        # by default, split on the reasoning_end_token and strip a reasoning_start_token from the start
        if self.reasoning_end_token is None or self.reasoning_end_token not in content:
            return content
        reasoning, content = content.rsplit(self.reasoning_end_token, 1)
        if self.reasoning_start_token is not None:
            reasoning = reasoning.strip().removeprefix(self.reasoning_start_token)
        return [ReasoningPart(content=reasoning.strip()), content.strip()]

    def parse_completion(self, completion: BaseCompletion) -> BaseCompletion:
        """
        Single-step parsing, if you prefer handling it all in one place. By default, calls :meth:`parse_tool_calls` and
        :meth:`parse_reasoning`.
        """
        # parse tool calls, merge if inner has tool calls
        completion.message.content, tool_calls = self.parse_tool_calls(completion.message.text)
        if completion.message.tool_calls and tool_calls:
            warnings.warn(
                f"Both the tool parser ({type(self).__name__}) and the wrapped engine's"
                f" ({type(self.engine).__name__}) completion returned tool calls. These will be concatenated, but"
                " may lead to unexpected behaviour! Make sure you are only parsing tool calls once.\n"
                f"Tool parser: {tool_calls}\nInner completion: {completion.message.tool_calls}",
                stacklevel=4,
            )
        completion.message.tool_calls = (completion.message.tool_calls or []) + tool_calls

        # parse reasoning
        completion.message.content = self.parse_reasoning(completion.message.content)
        return completion

    async def predict(self, messages, functions=None, **hyperparams) -> BaseCompletion:
        completion = await super().predict(messages, functions, **hyperparams)
        return self.parse_completion(completion)

    async def stream(self, messages, functions=None, **hyperparams):
        content_parts = []
        in_tool_call = False
        in_reasoning = self.reasoning_always_at_start
        inner_completion = None

        # make sure we yield the color if reasoning_always_at_start
        if self.show_reasoning_in_stream and self.reasoning_in_stream_color and self.reasoning_always_at_start:
            yield "\033[0;37m"

        # consume from the inner iterator, yielding as normal until we see a tool call or a completion
        async for elem in super().stream(messages, functions, **hyperparams):
            log.debug(f"Got stream element: {elem!r}")
            if isinstance(elem, str):
                content_parts.append(elem)

                # split by any special tokens and then process that way
                if self.splitter_re is None:
                    # generally this shouldn't happen since it only happens if we have no special tokens defined, but
                    # if so just yield and continue I guess
                    yield elem
                    continue

                for part in self.splitter_re.split(elem):
                    # by now, *part* must be either a special token in its entirety, empty str, or something to buffer
                    if not part:
                        continue

                    # special tokens
                    if self.tool_call_start_token is not None and part == self.tool_call_start_token:
                        in_tool_call = True
                        continue
                    if self.reasoning_start_token is not None and part == self.reasoning_start_token:
                        in_reasoning = True
                        if self.show_reasoning_in_stream and self.reasoning_in_stream_color:
                            yield "\033[0;37m"
                        continue
                    if self.tool_call_end_token is not None and part == self.tool_call_end_token:
                        in_tool_call = False
                        continue
                    if self.reasoning_end_token is not None and part == self.reasoning_end_token:
                        if self.show_reasoning_in_stream and self.reasoning_in_stream_color:
                            yield "\033[0m"
                        in_reasoning = False
                        continue

                    # content
                    if in_tool_call:
                        pass
                    elif in_reasoning:
                        if self.show_reasoning_in_stream:
                            yield part
                    else:
                        yield part
            else:
                # save the inner completion
                inner_completion = elem

        # we have consumed all the elements - construct a new completion
        # parse tool calls from the content (preserving inner tool calls if necessary)
        content = "".join(content_parts)
        log.debug(f"Content before parsing tool calls/reasoning: {content!r}")
        if inner_completion:
            yield self.parse_completion(inner_completion)
        else:
            completion = Completion(ChatMessage.assistant(content), prompt_tokens=None, completion_tokens=None)
            yield self.parse_completion(completion)


# backwards compatibility alias
BaseToolCallParser = BaseParser
