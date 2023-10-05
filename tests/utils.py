from kani import ChatMessage


def dict_at_least(outer: dict, inner: dict) -> bool:
    """Does the outer dict contain at least all kvs from the inner dict (checking dicts recursively)?"""
    for k, v in inner.items():
        if k not in outer:
            return False
        if not isinstance(outer[k], type(v)):
            return False
        # value: if it's a dict, recursive call; otherwise check equality
        if isinstance(v, dict):
            if not dict_at_least(outer[k], v):
                return False
        elif outer[k] != v:
            return False
    return True


def flatten_chatmessages(msgs: list[ChatMessage], between: str = "") -> str:
    """Return a string that is the concatenation of all contents of the chat messages."""
    return between.join((m.text or "") for m in msgs)
