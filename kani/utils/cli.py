"""The CLI utilities allow you to play with a chat session directly from a terminal."""

import asyncio
import logging
import os
import textwrap
from concurrent.futures import Future
from threading import Thread
from typing import AsyncIterable, overload

from kani import _optional
from kani.kani import Kani
from kani.models import ChatRole
from kani.streaming import StreamManager
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking


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
    stream: bool = True,
):
    """Async version of :func:`.chat_in_terminal`.
    Use in environments when there is already an asyncio loop running (e.g. Google Colab).
    """
    if os.getenv("KANI_DEBUG") is not None:
        logging.basicConfig(level=logging.DEBUG)
    elif os.getenv("KANI_DEBUG_LOGGERS") is not None:
        logging.basicConfig(level=logging.INFO)
        for logger in os.getenv("KANI_DEBUG_LOGGERS").split(","):
            logging.getLogger(logger).setLevel(logging.DEBUG)
    if verbose:
        echo = show_function_args = show_function_returns = True

    try:
        round_num = 0
        while round_num < rounds or not rounds:
            round_num += 1

            # get user query
            if not ai_first or round_num > 0:
                query = await ainput("USER: ")
                query = query_str = query.strip()

                # multimodal handling
                if _optional.has_multimodal_core:
                    # find @path/to/file.png parts and replace them with FileImageParts
                    query = await _optional.multimodal_cli.parts_from_cli_query(query)

                # stopword
                if stopword and query == stopword:
                    break

                # echo & multimodal echo
                if _optional.has_multimodal_core:
                    # IPython
                    if _optional.multimodal_cli._is_notebook:
                        _optional.multimodal_cli.display_media_ipython(query, show_text=echo)
                    else:
                        _optional.multimodal_cli.display_media(query, show_text=echo)
                elif echo:
                    print_width(query_str, width=width, prefix="USER: ")
            # print completion(s)
            else:
                query = None

            # print completion(s)
            if stream:
                async for stream in kani.full_round_stream(query):
                    # assistant
                    if stream.role == ChatRole.ASSISTANT:
                        await print_stream(stream, width=width, prefix="AI: ")
                        msg = await stream.message()
                        text = assistant_message_thinking(msg, show_args=show_function_args)
                        if text:
                            print_width(text, width=width, prefix="AI: ")
                    # function
                    elif stream.role == ChatRole.FUNCTION and show_function_returns:
                        msg = await stream.message()
                        print_width(msg.text, width=width, prefix="FUNC: ")
            # completions only
            else:
                async for msg in kani.full_round(query):
                    # assistant
                    if msg.role == ChatRole.ASSISTANT:
                        text = assistant_message_contents_thinking(msg, show_args=show_function_args)
                        print_width(text, width=width, prefix="AI: ")
                    # function
                    elif msg.role == ChatRole.FUNCTION and show_function_returns:
                        print_width(msg.text, width=width, prefix="FUNC: ")
    except (KeyboardInterrupt, asyncio.CancelledError):
        # we won't close the engine here since it's common enough that people close the session in colab
        # and if the process is closing then this will clean itself up anyway
        # await kani.engine.close()
        return


@overload
def chat_in_terminal(
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
    stream: bool = True,
): ...


def chat_in_terminal(kani: Kani, **kwargs):
    """Chat with a kani right in your terminal.

    Useful for playing with kani, quick prompt engineering, or demoing the library.

    If the environment variable ``KANI_DEBUG`` is set, debug logging will be enabled.

    If ``kani-multimodal-core`` is installed, you can send multimodal media to a compatible engine with a file path
    or URL after an ``@`` symbol (e.g. "Describe this image: @image.png").
    Use quotes (e.g. ``@"path/to/my image.png"``) for paths with spaces in their names.

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
    :param bool stream: Whether or not to print tokens as soon as they are generated by the model (default true).
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
    try:
        asyncio.run(chat_in_terminal_async(kani, **kwargs))
    except KeyboardInterrupt:
        return


# ===== format helpers =====
def format_width(msg: str, width: int = None, prefix: str = "") -> str:
    """
    Format the given message such that the width of each line is less than *width*.
    If *prefix* is provided, indents each line after the first by the length of the prefix.

    .. code-block: pycon
        >>> format_width("Hello world I am a potato", width=15, prefix="USER: ")
        '''\
        USER: Hello
              world I
              am a
              potato\
        '''
    """
    if not width:
        return prefix + msg
    out = []
    wrapper = textwrap.TextWrapper(width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix))
    lines = msg.splitlines()
    for line in lines:
        out.append(wrapper.fill(line))
        wrapper.initial_indent = wrapper.subsequent_indent
    return "\n".join(out)


async def format_stream(stream: StreamManager, width: int = None, prefix: str = "") -> AsyncIterable[str]:
    """
    Yield formatted tokens from a stream such that if concatenated, the width of each line is less than *width*.
    If *prefix* is provided, indents each line after the first by the length of the prefix.
    """
    prefix_len = len(prefix)
    line_indent = " " * prefix_len
    prefix_printed = False

    # print tokens until they overflow width then newline and indent
    line_len = prefix_len
    async for token in stream:
        # only print the prefix if the model actually yields anything
        if not prefix_printed:
            yield prefix
            prefix_printed = True

        # split by newlines
        for part in token.splitlines(keepends=True):
            # then do bookkeeping
            part_len = len(part)
            if width and line_len + part_len > width:
                yield f"\n{line_indent}"
                line_len = prefix_len

            # print the token
            yield part.rstrip("\r\n")
            line_len += part_len

            # print a newline if the token had one
            if part.endswith("\n"):
                yield f"\n{line_indent}"
                line_len = prefix_len


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
    print(format_width(msg, width, prefix))


async def print_stream(stream: StreamManager, width: int = None, prefix: str = ""):
    """
    Print tokens from a stream to the terminal, with the width of each line less than *width*.
    If *prefix* is provided, indents each line after the first by the length of the prefix.

    This is a helper function intended to be used with :meth:`.Kani.chat_round_stream` or
    :meth:`.Kani.full_round_stream`.
    """
    has_printed = False
    async for part in format_stream(stream, width, prefix):
        print(part, end="", flush=True)
        has_printed = True

    # newline at the end to flush if we printed anything
    if has_printed:
        print()


async def ainput(string: str) -> str:
    """input(), but async."""
    # doing just .to_thread causes problems when we ^C, so we need to launch our own daemon thread to handle reading
    # input
    future = Future()
    future.set_running_or_notify_cancel()

    def daemon():
        try:
            result = input(string)
        except Exception as e:
            future.set_exception(e)
        else:
            future.set_result(result)

    Thread(target=daemon, daemon=True).start()
    return await asyncio.wrap_future(future)


# ==== CLI engine defs ====
def chat_openai(model_id: str):
    from kani.engines.openai import OpenAIEngine

    return OpenAIEngine(model=model_id)


def chat_anthropic(model_id: str):
    from kani.engines.anthropic import AnthropicEngine

    return AnthropicEngine(model=model_id)


def chat_google(model_id: str):
    from kani.engines.google import GoogleAIEngine

    return GoogleAIEngine(model=model_id)


def chat_huggingface(model_id: str):
    from kani.engines.huggingface import HuggingEngine

    return HuggingEngine(model_id=model_id)


CLI_PROVIDER_MAP = {
    # openai
    "openai": chat_openai,
    "oai": chat_openai,
    # anthropic
    "anthropic": chat_anthropic,
    "ant": chat_anthropic,
    "claude": chat_anthropic,
    # google
    "google": chat_google,
    "g": chat_google,
    "gemini": chat_google,
    # huggingface
    "huggingface": chat_huggingface,
    "hf": chat_huggingface,
}


def create_engine_from_cli_arg(arg: str):
    provider, model_id = arg.split(":", 1)
    if provider not in CLI_PROVIDER_MAP:
        raise ValueError(f"Invalid model provider: {provider!r}. Valid options: {list(CLI_PROVIDER_MAP)}")
    return CLI_PROVIDER_MAP[provider](model_id)
