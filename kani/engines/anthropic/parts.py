from kani import MessagePart


# ===== Generic =====
class AnthropicUnknownPart(MessagePart):
    """
    A generic unknown response part from the server.

    This generally corresponds to an Anthropic-specific feature. The raw response data is accessible in ``data``,
    and will be sent back to the language model in future rounds correctly. Will not be sent to other engines.
    """

    type: str
    data: dict
    """The raw content of the part returned by the Anthropic API."""
