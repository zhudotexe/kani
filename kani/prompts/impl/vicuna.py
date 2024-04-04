"""Common builder for the Vicuna chat prompt."""

from kani.models import ChatRole
from kani.prompts.pipeline import PromptPipeline

VICUNA_PIPELINE = (
    PromptPipeline()
    .merge_consecutive(role=(ChatRole.USER, ChatRole.FUNCTION), sep="\n", out_role=ChatRole.USER)
    .merge_consecutive(role=ChatRole.ASSISTANT, sep=" ")
    .conversation_fmt(
        sep="\n",
        user_prefix="USER: ",
        assistant_prefix="ASSISTANT: ",
        assistant_suffix="</s>",
        assistant_suffix_if_last="",
        generation_suffix="ASSISTANT:"
    )
)  # fmt: skip
