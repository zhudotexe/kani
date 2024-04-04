import asyncio

from transformers import AutoTokenizer, TextStreamer


class AsyncTextIteratorStreamer(TextStreamer):
    """A HF streamer like the TextIteratorStreamer but using an async queue."""

    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: float = None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = asyncio.Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put_nowait(text)
        if stream_end:
            self.text_queue.put_nowait(self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await asyncio.wait_for(self.text_queue.get(), timeout=self.timeout)
        if value == self.stop_signal:
            raise StopAsyncIteration()
        else:
            return value
