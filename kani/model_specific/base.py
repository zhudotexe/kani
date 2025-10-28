import logging
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
        """
        super().__init__(*args, **kwargs)
        self.tool_call_start_token = tool_call_start_token
        self.tool_call_end_token = tool_call_end_token
        self.reasoning_start_token = reasoning_start_token
        self.reasoning_end_token = reasoning_end_token
        self.reasoning_always_at_start = reasoning_always_at_start
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

        # consume from the inner iterator, yielding as normal until we see a tool call or a completion
        async for elem in super().stream(messages, functions, **hyperparams):
            log.debug(f"Got stream element: {elem!r}")
            if isinstance(elem, str):
                content_parts.append(elem)
                # if we see the start of a tool call/reasoning, stop yielding and start buffering
                # noinspection DuplicatedCode
                if self.tool_call_start_token is not None and self.tool_call_start_token in elem:
                    if len(elem) > len(self.tool_call_start_token) and not in_reasoning:
                        yield elem[: elem.index(self.tool_call_start_token)]
                    in_tool_call = True
                if self.reasoning_start_token is not None and self.reasoning_start_token in elem:
                    if len(elem) > len(self.reasoning_start_token) and not in_tool_call:
                        yield elem[: elem.index(self.reasoning_start_token)]
                    in_reasoning = True

                # yield the string if not a special case
                if not (in_tool_call or in_reasoning):
                    yield elem

                # if we see the end of a tool call/reasoning, start yielding and stop buffering
                if self.tool_call_end_token is not None and self.tool_call_end_token in elem:
                    if len(elem) > len(self.tool_call_end_token) and not in_reasoning:
                        yield elem[elem.index(self.tool_call_end_token) + len(self.tool_call_end_token) :]
                    in_tool_call = False
                if self.reasoning_end_token is not None and self.reasoning_end_token in elem:
                    if len(elem) > len(self.reasoning_end_token) and not in_tool_call:
                        yield elem[elem.index(self.reasoning_end_token) + len(self.reasoning_end_token) :]
                    in_reasoning = False
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
