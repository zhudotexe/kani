"""The CLI utilities allow you to play with a chat session directly from a terminal."""
import asyncio

from kani.kani import Kani
from kani.models import ChatMessage


def _function_formatter(function_call: ChatMessage):
    return f"Thinking ({function_call.function_call.name})..."


async def _chat_in_terminal(kani: Kani):
    try:
        while True:
            query = input("USER: ")
            async for msg in kani.full_round(query, function_call_formatter=_function_formatter):
                print(f"AI: {msg}")
    except KeyboardInterrupt:
        await kani.engine.close()


def chat_in_terminal(kani: Kani):
    asyncio.run(_chat_in_terminal(kani))
