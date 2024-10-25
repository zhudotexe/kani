import os

import d20  # pip install d20

from kani import Kani, ai_function, chat_in_terminal
from kani.engines.openai import OpenAIEngine

engine = OpenAIEngine(os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")


class DiceKani(Kani):
    @ai_function
    def roll(self, dice: str):
        """Roll some dice or do math. Dice should be specified in the XdY format."""
        return d20.roll(dice).result


ai = DiceKani(engine)

if __name__ == "__main__":
    chat_in_terminal(ai)
