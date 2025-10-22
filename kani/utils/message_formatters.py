"""
A couple convenience formatters to customize :meth:`.Kani.full_round_str`.

You can pass any of these functions in with, e.g., ``Kani.full_round_str(..., message_formatter=all_message_contents)``.

"""

import itertools

from kani.models import ChatMessage, ChatRole
from kani.parts.reasoning import ReasoningPart


def all_message_contents(msg: ChatMessage):
    """Return the content of any message."""
    return msg.text


def assistant_message_contents(msg: ChatMessage, show_reasoning=True, color=True):
    """
    Return the content of any assistant message; otherwise don't return anything.

    :param show_reasoning: If True, include any ReasoningParts in the output.
    :param color: If True, the returned reasoning parts will be surrounded in ANSI codes to make it appear gray.
    """
    if msg.role != ChatRole.ASSISTANT:
        return

    if show_reasoning:
        text_parts = []
        for t, grp in itertools.groupby(msg.parts, lambda p: type(p)):
            if issubclass(t, ReasoningPart):
                content = "".join(p.content for p in grp)
                text_parts.append(f"\033[0;37m{content}\033[0m" if color else content)
            else:
                text_parts.append("".join(map(str, grp)))
        return "\n".join(text_parts)
    return msg.text or ""


def assistant_message_contents_thinking(msg: ChatMessage, show_args=False, show_reasoning=True, color=True):
    """
    Return the content of any assistant message, and "Thinking..." on function calls.

    You can use this in ``full_round_str`` by using a partial, e.g.:
    ``ai.full_round_str(..., message_formatter=functools.partial(assistant_message_contents_thinking, show_args=True))``

    :param show_args: If True, include the arguments to each function call.
    :param show_reasoning: If True, include any ReasoningParts in the output.
    :param color: If True, the returned reasoning parts will be surrounded in ANSI codes to make it appear gray.
    """
    if msg.role != ChatRole.ASSISTANT:
        return

    # text
    text = assistant_message_contents(msg, show_reasoning, color)

    # function calls
    if not msg.tool_calls:
        return text
    return f"{text}\n{assistant_message_thinking(msg, show_args)}".strip()


def assistant_message_thinking(msg: ChatMessage, show_args=False):
    """Return "Thinking..." on assistant messages with function calls, ignoring any content.

    This is useful if you are streaming the message's contents.

    If *show_args* is True, include the arguments to each function call.
    """
    if msg.role != ChatRole.ASSISTANT or not msg.tool_calls:
        return

    # with args: show a nice repr (e.g. `get_weather(location="San Francisco, CA", unit="fahrenheit")`)
    if show_args:
        parts = []
        for tc in msg.tool_calls:
            args = ", ".join(f"{kwarg}={v!r}" for kwarg, v in tc.function.kwargs.items())
            parts.append(f"{tc.function.name}({args})")
        called_functions = "; ".join(parts)
    # no args: just print the function name
    else:
        called_functions = "; ".join(tc.function.name for tc in msg.tool_calls)
    return f"Thinking... [{called_functions}]"
