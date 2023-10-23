"""The CLI utilities allow you to play with a chat session directly from a terminal."""

import asyncio
import logging
import os

from kani.kani import Kani
from kani.utils.message_formatters import assistant_message_contents_thinking


async def chat_in_terminal_async(kani: Kani, rounds: int = 0, stopword: str = None):
    """Async version of :func:`.chat_in_terminal`.
    Use in environments when there is already an asyncio loop running (e.g. Google Colab).
    """
    if os.getenv("KANI_DEBUG") is not None:
        logging.basicConfig(level=logging.DEBUG)

    try:
        round_num = 0
        while round_num < rounds or not rounds:
            round_num += 1
            query = input("USER: ").strip()
            if stopword and query == stopword:
                break
            async for msg in kani.full_round_str(query, message_formatter=assistant_message_contents_thinking):
                print(f"AI: {msg}")
    except KeyboardInterrupt:
        pass
    finally:
        await kani.engine.close()


def chat_in_terminal(kani: Kani, rounds: int = 0, stopword: str = None):
    """Chat with a kani right in your terminal.

    Useful for playing with kani, quick prompt engineering, or demoing the library.

    If the environment variable ``KANI_DEBUG`` is set, debug logging will be enabled.

    .. warning::

        This function is only a development utility and should not be used in production.

    :param rounds: The number of chat rounds to play (defaults to 0 for infinite).
    :param stopword: Break out of the chat loop if the user sends this message.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        try:
            # google colab comes with this pre-installed
            # let's try importing and patching the loop so that we can just use the normal asyncio.run call
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            print(
                f"WARNING: It looks like you're in an environment with a running asyncio loop (e.g. Google Colab).\nYou"
                f" should use `await chat_in_terminal_async(...)` instead or install `nest-asyncio`."
            )
            return
    asyncio.run(chat_in_terminal_async(kani, rounds=rounds, stopword=stopword))
