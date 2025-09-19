"""
Usage: python short-context.py org-id/model-id
"""

import logging
import sys

from kani import Kani, chat_in_terminal
from kani.engines.huggingface import HuggingEngine

logging.basicConfig(level=logging.INFO)

engine = HuggingEngine(sys.argv[-1], use_kv_cache=True, max_context_size=1024)
ai = Kani(engine)

if __name__ == "__main__":
    chat_in_terminal(ai, show_function_args=True, show_function_returns=True, stopword="!stop")
