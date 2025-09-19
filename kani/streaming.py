import asyncio
import contextlib
from collections.abc import AsyncIterable

from kani.engines.base import BaseCompletion, Completion
from kani.models import ChatMessage, ChatRole


class StreamManager:
    """
    This class is responsible for managing a stream returned by an engine. It should not be constructed manually.

    To consume tokens from a stream, use this class as so::

        # CHAT ROUND:
        stream = ai.chat_round_stream("What is the airspeed velocity of an unladen swallow?")
        async for token in stream:
            print(token, end="")
        msg = await stream.message()

        # FULL ROUND:
        async for stream in ai.full_round_stream("What is the airspeed velocity of an unladen swallow?")
            async for token in stream:
                print(token, end="")
            msg = await stream.message()

    After a stream finishes, its contents will be available as a :class:`.ChatMessage`. You can retrieve the final
    message or :class:`.BaseCompletion` with::

        msg = await stream.message()
        completion = await stream.completion()

    The final :class:`.ChatMessage` may contain non-yielded tokens (e.g. a request for a function call). If the final
    message or completion is requested before the stream is iterated over, the stream manager will consume the entire
    stream.

    .. tip::
        For compatibility and ease of refactoring, awaiting the stream itself will also return the message, i.e.:

        .. code-block:: python

            msg = await ai.chat_round_stream("What is the airspeed velocity of an unladen swallow?")

        (note the ``await`` that is not present in the above examples).
    """

    def __init__(
        self, stream_iter: AsyncIterable[str | BaseCompletion], role: ChatRole, *, after=None, lock: asyncio.Lock = None
    ):
        """
        :param stream_iter: The async iterable that generates elements of the stream.
        :param role: The role of the message that will be returned eventually.
        :param after: A coro to call with the generated completion as its argument after the stream is fully consumed.
        :param lock: A lock to hold for the duration of the stream run.
        """
        self.role = role
        """The role of the message that this stream will return."""

        # private
        self._stream_iter = stream_iter
        self._after = after
        self._lock = lock if lock is not None else contextlib.nullcontext()

        # results
        self._completion = None  # the final completion result
        self._awaited = False  # whether or not this stream has already been consumed
        self._finished = asyncio.Event()  # whether or not the stream has finished

    # ==== stream components ====
    async def _stream_impl_outer(self):
        # simple wrapper to lock the lock too, but *async*
        async with self._lock:
            async for elem in self._stream_impl():
                yield elem

    async def _stream_impl(self):
        """
        Wrap the underlying stream iterable to handle mixed yield types and build completion when the stream finishes
        without setting the completion.
        """
        yielded_tokens = []

        # for each token or completion yielded by the engine,
        async for elem in self._stream_iter:
            if self._completion is not None:
                raise RuntimeError(
                    "Expected `BaseCompletion` to be final yield of stream iterable but got another value after!"
                )

            # re-yield if str
            if isinstance(elem, str):
                yield elem
                yielded_tokens.append(elem)
            # save if completion
            elif isinstance(elem, BaseCompletion):
                self._completion = elem
            # panic otherwise
            else:
                raise TypeError(
                    "Expected yielded value from stream iterable to be `str` or `BaseCompletion` but got"
                    f" {type(elem)!r}!"
                )

        # if the stream is complete but we did not get a completion, we'll construct one here as the concatenation
        # of all the yielded tokens
        if self._completion is None:
            content = "".join(yielded_tokens)
            self._completion = Completion(message=ChatMessage(role=self.role, content=content.strip()))

        # run the callback, if any
        if self._after is not None:
            await self._after(self._completion)

        # allow anything waiting on the stream to finish to progress
        self._finished.set()

    def __aiter__(self) -> AsyncIterable[str]:
        """Iterate over tokens yielded from the engine."""
        # enforce that it can only be iterated over once
        if self._awaited:
            raise RuntimeError(
                "This stream has already been consumed. If you are consuming both the stream and the final Completion,"
                " make sure you iterate over the stream first."
            )
        self._awaited = True
        # delegate to a wrapper for an async context
        return self._stream_impl_outer()

    # ==== final result getters ====
    def __await__(self):
        """Awaiting the StreamManager is equivalent to awaiting :meth:`message`."""
        return self.message().__await__()

    async def completion(self) -> BaseCompletion:
        """Get the final :class:`.BaseCompletion` generated by the model."""
        # if we are getting the completion but no one has consumed our stream yet, just dummy do it so we build
        # the completion
        if not self._awaited:
            async for _ in self:
                pass

        # otherwise, wait for the stream to be complete then return the saved completion
        await self._finished.wait()
        return self._completion

    async def message(self) -> ChatMessage:
        """Get the final :class:`.ChatMessage` generated by the model."""
        completion = await self.completion()
        return completion.message


class DummyStream(StreamManager):
    """Function calling helper: we already have the message."""

    def __init__(self, message: ChatMessage):
        # init a dummy iterable
        async def _iter():
            if message.content is not None:
                yield message.text
            yield Completion(message)

        super().__init__(_iter(), role=message.role)
        self._message = message

    async def message(self) -> ChatMessage:
        return self._message
