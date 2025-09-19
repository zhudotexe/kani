import logging
import re

from kani.engines.huggingface.chat_template_pipeline import ChatTemplatePromptPipeline, hf_tool_use_keys
from kani.models import FunctionCall, MessagePart, ToolCall
from kani.parts import ReasoningPart
from .base import BaseToolCallParser

SPECIAL_TOKEN_REGEX = re.compile(r"<\|(?P<type>\w+)\|>")
SPECIAL_TOKEN_REGEX_2 = re.compile(r"(<\|\w+\|>)")
TO_REGEX = re.compile(rf"to=([^\s<]+)")
CHANNEL_REGEX = re.compile(r"<\|channel\|>(\w+)")
ASST_MSG_REGEX = re.compile(
    r"(<\|start\|>(?P<role>\w+))?"
    r"(?P<header>.*?)"
    r"<\|message\|>(?P<content>.*?)"
    r"(<\|end\|>|<\|return\|>|<\|call\|>|$)",
    re.DOTALL,
)
log = logging.getLogger(__name__)


# ===== PROMPT PIPELINE =====
# TODO maybe merge this into the base chat template pipeline if more people adopt this?
# if so we can remove the pipeline from model_specific.__init__
def _gptoss_chat_template_keys(message):
    """Sets the thinking key based on any ReasoningPart's in the message."""
    reasoning_parts = [p for p in message.parts if isinstance(p, ReasoningPart)]
    reasoning = "\n".join(r.content for r in reasoning_parts)
    keys = hf_tool_use_keys(message)
    if reasoning:
        keys["thinking"] = reasoning
    return keys


def build_gptoss_prompt_pipeline(tokenizer, **kwargs):
    """We just extend the chat template pipeline with a bit of extra code to make sure the thinking key is set"""
    return ChatTemplatePromptPipeline(tokenizer, **kwargs).conversation_dict(additional_keys=_gptoss_chat_template_keys)


# ===== OUTPUT PARSER =====
class GPTOSSParser(BaseToolCallParser):
    r"""
    Automatically handles the parsing of GPT-OSS reasoning segments and tool calls.

    Reasoning segments are returned as :class:`.ReasoningPart`\ s.
    """

    def __init__(self, *args, show_reasoning_in_stream=False, **kwargs):
        """
        :param show_reasoning_in_stream: Whether reasoning tokens should be yielded during streams. By default, only
            non-reasoning tokens will be yielded, and reasoning tokens will be included in a :class:`.ReasoningPart` in
            the final :class:`.ChatMessage`.
        """
        super().__init__(*args, tool_call_start_token=None, tool_call_end_token=None, **kwargs)
        self.show_reasoning_in_stream = show_reasoning_in_stream

    # state machine for stream on special token, regex for parse
    def parse_tool_calls(self, content: str) -> tuple[list[MessagePart | str], list[ToolCall]]:
        log.debug(f"PARSING MSG: {content}")
        parts = []
        tcs = []

        for match in ASST_MSG_REGEX.finditer(content):
            log.debug(f"PART: {match[0]}")
            header = match["header"]
            channel = c[1] if (c := CHANNEL_REGEX.search(header)) else None
            to = t[1] if (t := TO_REGEX.search(header)) else None
            content = match["content"]

            if to and to.startswith("functions."):
                tcs.append(
                    ToolCall.from_function_call(FunctionCall(name=to.removeprefix("functions."), arguments=content))
                )
            elif channel == "analysis":
                parts.append(ReasoningPart(content=content))
            else:
                parts.append(content)

        return parts, tcs

    async def stream(self, messages, functions=None, **hyperparams):
        state = _GPTOSSStreamState(show_reasoning=self.show_reasoning_in_stream)
        async for elem in super().stream(messages, functions, **hyperparams):
            if isinstance(elem, str):
                for tokenlike in SPECIAL_TOKEN_REGEX_2.split(elem):
                    if not tokenlike:
                        continue
                    to_yield = state.feed(tokenlike)
                    if to_yield:
                        yield to_yield
            else:
                yield elem  # probably the inner completion


class _GPTOSSStreamState:
    def __init__(self, show_reasoning):
        # the last seen special token, e.g. "start", "channel", "constrain", "message"
        # end sets this to None
        # https://cookbook.openai.com/articles/openai-harmony#special-tokens
        self.show_reasoning = show_reasoning
        self.state = None
        self.channel = None
        self.to = None
        self.buf = []

    def feed(self, part: str):
        # update the state machine
        # new state
        if match := SPECIAL_TOKEN_REGEX.fullmatch(part):
            self.transition_states(match["type"])
        # in message state and part is visible to user: yield it
        elif self.is_visible_to_user():
            return part
        # default: keep buffering
        else:
            self.buf.append(part)
        return None

    def transition_states(self, new_state: str):
        # we are about to transition states, handle the last state
        buf_str = "".join(self.buf)
        # check for to=... (in states None, start, or channel)
        if self.state in (None, "start", "channel") and (match := TO_REGEX.search(buf_str)):
            self.to = match[1]
        # check if we are finishing a channel
        if self.state == "channel":
            self.channel, *_ = buf_str.split(" ", 1)

        # if our new state is "end", clear the state
        if new_state == "end":
            self.channel = None
            self.to = None
        self.buf.clear()

        self.state = new_state
        log.debug(f"STREAM NEW STATE: {self!r}")

    def is_visible_to_user(self):
        # the content is visible to the user IFF:
        # - state is "message"
        # - channel is "final" or "commentary"
        # - to is None
        if self.show_reasoning:
            return self.state == "message" and self.channel in ("final", "commentary", "analysis") and self.to is None
        return self.state == "message" and self.channel in ("final", "commentary") and self.to is None

    def __repr__(self):
        return f"<_GPTOSSStreamState {self.state=} {self.channel=} {self.to=} {self.buf=}>"
