"""The CLI utilities allow you to play with a chat session directly from a terminal."""
import asyncio

from chatterbox import ChatterboxWithFunctions
from chatterbox.models import ChatMessage


def _function_formatter(function_call: ChatMessage):
    return f"Thinking ({function_call.function_call.name})..."


async def _chat_in_terminal(chatterbox: ChatterboxWithFunctions):
    try:
        while True:
            query = input("USER: ")
            async for msg in chatterbox.full_round(query, function_call_formatter=_function_formatter):
                print(f"AI: {msg}")
    except KeyboardInterrupt:
        await chatterbox.client.close()


def chat_in_terminal(chatterbox):
    asyncio.run(_chat_in_terminal(chatterbox))
