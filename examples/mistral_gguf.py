"""
A standalone Mistral-7B-Instruct file using llama.cpp.
"""

from kani import Kani, chat_in_terminal
from kani.engines.llamacpp import LlamaCppEngine

engine = LlamaCppEngine(repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", filename="*.Q4_K_M.gguf")

ai = Kani(engine)

if __name__ == "__main__":
    chat_in_terminal(ai)
