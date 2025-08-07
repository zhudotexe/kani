import logging
import warnings
from abc import ABC

from kani.engines import Completion, WrapperEngine
from kani.engines.base import BaseCompletion
from kani.models import ChatMessage, ToolCall

log = logging.getLogger(__name__)


class BaseToolCallParser(WrapperEngine, ABC):
    """
    Abstract base class for tool call parsers.

    To implement your own tool call parser, subclass this class and:

    * implement ``parse_tool_calls(self, content: str) -> tuple[str, list[ToolCall]]``
    * pass default values of ``tool_call_start_token`` and ``tool_call_end_token`` to ``super().__init__(...)``

    This class will handle calling the parser and interrupting streams when tool calls are detected.
    """

    def __init__(self, *args, tool_call_start_token: str | None, tool_call_end_token: str | None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_call_start_token = tool_call_start_token
        self.tool_call_end_token = tool_call_end_token
        # a moderate hack; globally save that we have initialized some parser
        from kani import model_specific

        model_specific._has_initialized_model_specific_parser = True

    def parse_tool_calls(self, content: str) -> tuple[str, list[ToolCall]]:
        """Given the string completion of the model, return the content without tool calls and the parsed tool calls."""
        raise NotImplementedError

    async def predict(self, messages, functions=None, **hyperparams) -> BaseCompletion:
        completion = await super().predict(messages, functions, **hyperparams)

        # if we have tools, parse them
        if functions:
            completion.message.content, completion.message.tool_calls = self.parse_tool_calls(completion.message.text)

        return completion

    async def stream(self, messages, functions=None, **hyperparams):
        content_parts = []
        in_tool_call = False
        inner_completion = None

        # consume from the inner iterator, yielding as normal until we see a tool call or a completion
        async for elem in super().stream(messages, functions, **hyperparams):
            log.debug(f"Got stream element: {elem!r}")
            if isinstance(elem, str):
                content_parts.append(elem)
                # if we see the start of a tool call, stop yielding and start buffering
                if self.tool_call_start_token is not None and self.tool_call_start_token in elem:
                    if len(elem) > len(self.tool_call_start_token):
                        yield elem[: elem.index(self.tool_call_start_token)]
                    in_tool_call = True
                # if we see the end of a tool call, start yielding and stop buffering
                if self.tool_call_end_token is not None and self.tool_call_end_token in elem:
                    if len(elem) > len(self.tool_call_end_token):
                        yield elem[elem.index(self.tool_call_end_token) + len(self.tool_call_end_token) :]
                    in_tool_call = False
                # otherwise yield the string
                if not in_tool_call:
                    yield elem
            else:
                # save the inner completion
                inner_completion = elem

        # we have consumed all the elements - construct a new completion
        # parse tool calls from the content (preserving inner tool calls if necessary)
        content = "".join(content_parts)
        log.debug(f"Content before parsing tool calls: {content!r}")
        content, tool_calls = self.parse_tool_calls(content)
        if inner_completion:
            if inner_completion.message.tool_calls and tool_calls:
                warnings.warn(
                    f"Both the tool parser ({type(self).__name__}) and the wrapped engine's"
                    f" ({type(self.engine).__name__}) completion returned tool calls. These will be concatenated, but"
                    " may lead to unexpected behaviour! Make sure you are only parsing tool calls once.\n"
                    f"Tool parser: {tool_calls}\nInner completion: {inner_completion.message.tool_calls}",
                    stacklevel=3,
                )
            tool_calls = (inner_completion.message.tool_calls or []) + tool_calls
            prompt_tokens = inner_completion.prompt_tokens
            completion_tokens = inner_completion.completion_tokens
        else:
            prompt_tokens = None
            completion_tokens = None
        yield Completion(
            ChatMessage.assistant(content, tool_calls=tool_calls),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
