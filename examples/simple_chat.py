import os

from kani import Kani, chat_in_terminal
from kani.engines import OpenAIEngine

engine = OpenAIEngine(os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
ai = Kani(engine)

if __name__ == "__main__":
    chat_in_terminal(ai)
