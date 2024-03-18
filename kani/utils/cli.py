"""The CLI utilities allow you to play with a chat session directly from a terminal."""

import asyncio
import logging
import os
import textwrap

from kani.kani import Kani
from kani.models import ChatRole
from kani.utils.message_formatters import assistant_message_contents_thinking


async def chat_in_terminal_async(
    kani: Kani,
    *,
    rounds: int = 0,
    stopword: str = None,
    echo: bool = False,
    ai_first: bool = False,
    width: int = None,
    show_function_args: bool = False,
    show_function_returns: bool = False,
    verbose: bool = False,
):
    """Async version of :func:`.chat_in_terminal`.
    Use in environments when there is already an asyncio loop running (e.g. Google Colab).
    """
    if os.getenv("KANI_DEBUG") is not None:
        logging.basicConfig(level=logging.DEBUG)
    if verbose:
        echo = show_function_args = show_function_returns = True

    try:
        round_num = 0
        while round_num < rounds or not rounds:
            round_num += 1

            # get user query
            if not ai_first or round_num > 0:
                query = input("USER: ").strip()
                if echo:
                    print_width(query, width=width, prefix="USER: ")
                if stopword and query == stopword:
                    break
            else:
                query = None

            # print completion(s)
            async for msg in kani.full_round(query):
                # assistant
                if msg.role == ChatRole.ASSISTANT:
                    text = assistant_message_contents_thinking(msg, show_args=show_function_args)
                    print_width(text, width=width, prefix="AI: ")
                # function
                elif msg.role == ChatRole.FUNCTION and show_function_returns:
                    print_width(msg.text, width=width, prefix="FUNC: ")
    except KeyboardInterrupt:
        pass
    finally:
        await kani.engine.close()


def chat_in_terminal(kani: Kani, **kwargs):
    """Chat with a kani right in your terminal.

    Useful for playing with kani, quick prompt engineering, or demoing the library.

    If the environment variable ``KANI_DEBUG`` is set, debug logging will be enabled.

    .. warning::

        This function is only a development utility and should not be used in production.

    :param int rounds: The number of chat rounds to play (defaults to 0 for infinite).
    :param str stopword: Break out of the chat loop if the user sends this message.
    :param bool echo: Whether to echo the user's input to stdout after they send a message (e.g. to save in interactive
        notebook outputs; default false)
    :param bool ai_first: Whether the user should send the first message (default) or the model should generate a
        completion before prompting the user for a message.
    :param int width: The maximum width of the printed outputs (default unlimited).
    :param bool show_function_args: Whether to print the arguments the model is calling functions with for each call
        (default false).
    :param bool show_function_returns: Whether to print the results of each function call (default false).
    :param bool verbose: Equivalent to setting ``echo``, ``show_function_args``, and ``show_function_returns`` to True.
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
    asyncio.run(chat_in_terminal_async(kani, **kwargs))


def print_width(msg: str, width: int = None, prefix: str = ""):
    """
    Print the given message such that the width of each line is less than *width*.
    If *prefix* is provided, indents each line after the first by the length of the prefix.

    .. code-block: pycon
        >>> print_width("Hello world I am a potato", width=15, prefix="USER: ")
        USER: Hello
              world I
              am a
              potato
    """
    if not width:
        print(prefix + msg)
        return
    out = []
    wrapper = textwrap.TextWrapper(width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix))
    lines = msg.splitlines()
    for line in lines:
        out.append(wrapper.fill(line))
        wrapper.initial_indent = wrapper.subsequent_indent
    print("\n".join(out))
