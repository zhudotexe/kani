"""Vicuna (https://huggingface.co/lmsys/vicuna-7b-v1.3) is a language model based on LLaMA that is fine-tuned for chat.

This example shows how you can use kani to run a language model on your own machine! See the source code of
:class:`.VicunaEngine` for implementation details.
"""

from kani import Kani, chat_in_terminal
from kani.engines.huggingface.vicuna import VicunaEngine

engine = VicunaEngine()
ai = Kani(
    engine,
    system_prompt=(
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed,"
        " and polite answers to the user's questions."
    ),
)
if __name__ == "__main__":
    chat_in_terminal(ai)
