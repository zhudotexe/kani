"""Example from the Customization docs.

This example shows how to override kani's chat history handler to log every message to a JSONL file.
"""

import os

from kani import Kani, chat_in_terminal
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAIEngine(api_key, model="gpt-4o-mini")


class LogMessagesKani(Kani):
    # You can override __init__ and track kani-specific state:
    # in this example we keep track of the file we're logging to.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = open("kani-log.jsonl", "w")

    async def add_to_history(self, message):
        await super().add_to_history(message)
        self.log_file.write(message.model_dump_json())
        self.log_file.write("\n")


ai = LogMessagesKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai)
    ai.log_file.close()
