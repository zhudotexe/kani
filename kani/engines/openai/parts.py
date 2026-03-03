from kani import MessagePart, ReasoningPart


class OpenAIUnknownPart(MessagePart):
    """
    A generic unknown response part from the server.

    This generally corresponds to an OpenAI-specific feature. The raw response data is accessible in ``data``,
    and will be sent back to the language model in future rounds correctly. Will not be sent to other engines.
    """

    type: str
    data: dict
    """The raw content of the part returned by the OpenAI API."""


class OpenAIReasoningPart(ReasoningPart):
    """
    An OpenAI-specific reasoning part.

    This includes additional metadata specific to the OpenAI API that is used for multi-turn reasoning.
    """

    id: str
    encrypted_content: str
