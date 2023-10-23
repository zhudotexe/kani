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
        if msg.function_call and text:
            return f"{text}\n    Thinking ({msg.function_call.name})..."
        elif msg.function_call:
            return f"Thinking ({msg.function_call.name})..."
        return text
