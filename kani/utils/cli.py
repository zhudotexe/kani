"""The CLI utilities allow you to play with a chat session directly from a terminal."""
import asyncio

from kani.kani import Kani
from kani.models import ChatMessage


def _function_formatter(message: ChatMessage):
    return f"Thinking ({message.function_call.name})..."


async def _chat_in_terminal(kani: Kani, rounds: int = 0):
    try:
        round_num = 0
        while round_num < rounds or not rounds:
            round_num += 1
            query = input("USER: ")
            async for msg in kani.full_round_str(query, function_call_formatter=_function_formatter):
                print(f"AI: {msg}")
    finally:
        if not rounds:
            await kani.engine.close()


def chat_in_terminal(kani: Kani, rounds: int = 0):
    """Chat with a kani right in your terminal.

    Useful for playing with kani, quick prompt engineering, or demoing the library.

    .. warning::

        This function is only a development utility and should not be used in production.

    :param rounds: The number of chat rounds to play (defaults to 0 for infinite).
    """
    asyncio.run(_chat_in_terminal(kani, rounds))
