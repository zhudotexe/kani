"""
A couple convenience formatters to customize :meth:`.Kani.full_round_str`.

You can pass any of these functions in with, e.g., ``Kani.full_round_str(..., message_formatter=all_message_contents)``.

"""

from kani.models import ChatMessage, ChatRole


def all_message_contents(msg: ChatMessage):
    """Return the content of any message."""
    return msg.text


def assistant_message_contents(msg: ChatMessage):
    """Return the content of any assistant message; otherwise don't return anything."""
    if msg.role == ChatRole.ASSISTANT:
        return msg.text


def assistant_message_contents_thinking(msg: ChatMessage, show_args=False):
    """Return the content of any assistant message, and "Thinking..." on function calls.

    If *show_args* is True, include the arguments to each function call.
    You can use this in ``full_round_str`` by using a partial, e.g.:
    ``ai.full_round_str(..., message_formatter=functools.partial(assistant_message_contents_thinking, show_args=True))``
    """
    if msg.role != ChatRole.ASSISTANT:
        return

    # text
    text = msg.text or ""

    # function calls
    if not msg.tool_calls:
        function_calls = ""
    else:
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
        function_calls = f"\nThinking... [{called_functions}]"

    return (text + function_calls).strip()
