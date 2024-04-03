"""Common builder for the LLaMAv2-chat prompt.

This file is responsible for implementing the common non-strict ChatMessage to tokens translation, while handling
the nuance of the INST and SYS tokens as best as possible.
"""

from kani.models import ChatRole
from kani.prompts.pipeline import PromptPipeline

LLAMA2_PIPELINE = (
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
)  # fmt: skip
