"""LLaMA v2 (https://ai.meta.com/llama/) is a language model released by Meta that has variants fine-tuned for chat.

This example shows how you can use kani to run a language model on your own machine! See the source code of
:class:`.LlamaEngine` for implementation details.
"""

from kani import Kani, chat_in_terminal
from kani.engines.huggingface.llama2 import LlamaEngine

engine = LlamaEngine(use_auth_token=True)
ai = Kani(
    engine,
    system_prompt=(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
        " Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
        " Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not"
        " make any sense, or is not factually coherent, explain why instead of answering something not correct. If you"
        " don't know the answer to a question, please don't share false information."
    ),
)

if __name__ == "__main__":
    chat_in_terminal(ai)
