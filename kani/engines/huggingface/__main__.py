"""
For internal testing:
python -m kani.engines.huggingface org-id/model-id
Equivalent to initializing a HuggingEngine and calling chat_in_terminal.
"""

import sys

from kani import Kani, chat_in_terminal
from kani.engines.huggingface import HuggingEngine


def basic_chat_with_model_id(model_id: str):
    engine = HuggingEngine(model_id)
    ai = Kani(engine)
    chat_in_terminal(ai)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m kani.engines.huggingface <org-id/model-id>")
    basic_chat_with_model_id(sys.argv[1])
