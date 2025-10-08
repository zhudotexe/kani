from kani.models import MessagePart


class TextPart(MessagePart):
    """
    A MessagePart containing some text content. Generally using a :class:`str` is preferred; only use this MessagePart
    when a certain model requires storing additional metadata (in ``extra``) alongside the text content.
    """

    content: str

    def __str__(self):
        return self.content
