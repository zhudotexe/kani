import functools
from dataclasses import dataclass

from kani.models import ChatMessage


@dataclass
class PipelineExample:
    id: str  # for deduplication
    desc: str
    priority: int  # lower comes first
    example: list[ChatMessage]


def natural_join(elems: list[str], sep: str):
    sep = f" {sep} "
    if len(elems) < 3:
        return sep.join(elems)
    return ", ".join(elems[:-1]) + f",{sep}{elems[-1]}"


# we define all the examples as functions to save memory if the user never uses explain()
@functools.cache
def basic_conversation():
    return PipelineExample(
        id="basic",
        desc="A basic conversation including a system prompt, one full user-assistant turn, and a final user query.",
        priority=0,
        example=[
            ChatMessage.system("You are a helpful assistant."),
            ChatMessage.user("Hello there."),
            ChatMessage.assistant("Hi! How can I help?"),
            ChatMessage.user("What is the airspeed velocity of an unladen swallow?"),
        ],
    )
