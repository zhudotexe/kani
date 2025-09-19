"""
This example shows how to use kani's main entrypoints by setting up an async context.

In an experimental or production setting, you would likely load in data or accept user input through your own
mechanisms, rather than ``input()``. This example shows how the synchronous ``chat_in_terminal`` works under the hood
(with some simplifications, of course).

If you haven't already, go take a look at `1_quickstart.py` for an explanation of how to set up kani first.
"""

import asyncio

from kani import Kani
from kani.engines.openai import OpenAIEngine

api_key = "sk-..."
engine = OpenAIEngine(api_key, model="gpt-4o-mini")
ai = Kani(engine, system_prompt="You are a helpful assistant.")


# define your function normally, using `async def` instead of `def`
async def chat_with_kani():
    while True:
        user_message = input("USER: ")
        # now, you can use `await` to call kani's async methods
        message = await ai.chat_round_str(user_message)
        print("AI:", message)


# use `asyncio.run` to call your async function to start the program
asyncio.run(chat_with_kani())
