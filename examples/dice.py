import os

import d20  # pip install d20

from chatterbox import ChatterboxWithFunctions, ai_function
from chatterbox.engines import OpenAIClient
from chatterbox.utils.cli import chat_in_terminal

client = OpenAIClient(os.getenv("OPENAI_API_KEY"))


class DiceChatterbox(ChatterboxWithFunctions):
    @ai_function
    async def roll(self, dice: str):
        """Roll some dice or do math. Dice should be specified in the XdY format."""
        return d20.roll(dice).result


box = DiceChatterbox(client)

if __name__ == "__main__":
    chat_in_terminal(box)
