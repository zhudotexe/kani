import os

from kani import ChatterboxWithFunctions
from kani.engines.openai.client import OpenAIClient
from kani.utils.cli import chat_in_terminal

client = OpenAIClient(os.getenv("OPENAI_API_KEY"))
box = ChatterboxWithFunctions(client)

if __name__ == "__main__":
    chat_in_terminal(box)
