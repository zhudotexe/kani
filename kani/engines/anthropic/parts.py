from kani import MessagePart, ReasoningPart


class AnthropicUnknownPart(MessagePart):
    """
    A generic unknown response part from the server.

    This generally corresponds to an Anthropic-specific feature. The raw response data is accessible in ``data``,
    and will be sent back to the language model in future rounds correctly. Will not be sent to other engines.
    """

    type: str
    data: dict
    """The raw content of the part returned by the Anthropic API."""


class AnthropicThinkingPart(ReasoningPart):
    """
    An Anthropic-specific thinking part.

    This includes additional metadata specific to the Anthropic API that is used for multi-turn thinking.
    """

    signature: str
