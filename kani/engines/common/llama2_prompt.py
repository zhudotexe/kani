"""Common builder for the LLaMAv2-chat prompt.

This file is responsible for implementing the common non-strict ChatMessage to tokens translation, while handling
the nuance of the INST and SYS tokens as best as possible.
"""
from typing import Callable

from kani.models import ChatMessage, ChatRole

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def build(messages: list[ChatMessage], tokenize: Callable[[str], list[int]], eos_token_id: int = 2) -> list[int]:
    tokens = []
    prompt_buf = []  # parts of the user-assistant pair
    for message in messages:
        if message.role == ChatRole.USER:
            prompt_buf.append(f"{B_INST} {message.content} {E_INST}")
        elif message.role == ChatRole.ASSISTANT:
            prompt_buf.append(f" {message.content} ")
            # turn the current round into tokens
            prompt_round = "".join(prompt_buf)
            # hack: if we see a " {E_INST}{B_INST} " we should replace it with empty string
            # (it happens immediately after a system + user message)
            prompt_round.replace(f" {E_INST}{B_INST} ", "")
            tokens.extend(tokenize(prompt_round))  # assumption: tokenize() adds the BOS token but not the EOS token
            tokens.append(eos_token_id)
            prompt_buf.clear()
        else:
            prompt_buf.append(f"{B_INST} {B_SYS}{message.content}{E_SYS} {E_INST}")
    # flush rest of prompt buffer (probably a user message) into tokens
    if prompt_buf:
        tokens.extend(tokenize("".join(prompt_buf)))
    return tokens
