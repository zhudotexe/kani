from kani.models import MessagePart


class ReasoningPart(MessagePart):
    """
    A long CoT that should not be shown to the user (e.g. GPT-OSS, Anthropic extended thinking, Deepseek R1).

    When using a low-level text engine (e.g., :class:`.HuggingEngine`), these parts will not be automatically extracted.
    Use a parser instead (e.g., :class:`.GPTOSSParser` for GPT-OSS).
    """

    content: str
    """The reasoning content."""

    def __str__(self):
        """Reasoning content is hidden by default for models that don't explicitly request it"""
        return ""
