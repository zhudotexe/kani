"""
Example from Advanced Usage docs.

This example shows how to use the MessagePart API to allow tools to return multimodal content.

You should be familiar with the concept of Engines and Message Parts before trying to understand this code.
"""

from kani import Kani, ai_function, chat_in_terminal
from kani.ext.multimodal_core import ImagePart


class MyKani(Kani):
    @ai_function()
    async def get_image(self, url: str):
        """Download the image at a certain URL, and view it."""
        return [await ImagePart.from_url(url)]


engine = ...  # well, most LLMs can't yet actually take images as tool responses, but Kani supports it!
ai = MyKani(engine)

# Try this prompt:
# What is the image at https://raw.githubusercontent.com/zhudotexe/kani-multimodal-core/main/tests/data/test.png an image of?
if __name__ == "__main__":
    chat_in_terminal(ai)
