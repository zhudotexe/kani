"""Common builder for the LLaMAv2-chat prompt.

This file is responsible for implementing the common non-strict ChatMessage to tokens translation, while handling
the nuance of the INST and SYS tokens as best as possible.
"""

import itertools
from typing import Callable, Iterable

from kani.models import ChatMessage, ChatRole

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def build(
    messages: list[ChatMessage], tokenize: Callable[[str], list[int]], bos_token_id: int = 1, eos_token_id: int = 2
) -> list[int]:
    """Build the tokens for a list of messages. `tokenize` should tokenize a str without special tokens."""
    tokens = []
    for content, bos, eos in build_str(messages):
        if bos:
            tokens.append(bos_token_id)
        tokens.extend(tokenize(content))
        if eos:
            tokens.append(eos_token_id)
    return tokens


def build_str(messages: list[ChatMessage]) -> Iterable[tuple[str, bool, bool]]:
    """Given a list of messages, yield a list of pairs of (content string, bos, eos)."""
    # combine consecutive instruction messages and non-instruction messages
    for is_inst, role_messages in itertools.groupby(
        messages, key=lambda m: m.role == ChatRole.USER or m.role == ChatRole.SYSTEM
    ):
        # get content within tags (if any)
        content = []
        for message in role_messages:
            if message.role == ChatRole.SYSTEM:
                content.append(f"{B_SYS}{message.text}{E_SYS}")
            else:
                content.append(message.text)
        # if the content is an instruction, return it wrapped in inst tags; otherwise don't
        content_str = "".join(content)
        if is_inst:
            yield f"{B_INST} {content_str} {E_INST}", True, False
        else:
            yield f" {content_str} ", False, True
