import os

from chatterbox import ChatterboxWithFunctions
from chatterbox.engines.openai.client import OpenAIClient
from chatterbox.utils.cli import chat_in_terminal

client = OpenAIClient(os.getenv("OPENAI_API_KEY"))
box = ChatterboxWithFunctions(client)

if __name__ == "__main__":
    chat_in_terminal(box)
