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


def assistant_message_contents_thinking(msg: ChatMessage):
    """Return the content of any assistant message, and "Thinking..." on function calls."""
    if msg.role == ChatRole.ASSISTANT:
        text = msg.text
        if msg.tool_calls and text:
            called_functions = ", ".join(tc.function.name for tc in msg.tool_calls)
            return f"{text}\n    Thinking ({called_functions})..."
        elif msg.tool_calls:
            called_functions = ", ".join(tc.function.name for tc in msg.tool_calls)
            return f"Thinking ({called_functions})..."
        return text
