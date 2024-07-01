import abc
import warnings
from collections.abc import AsyncIterable

from kani.ai_function import AIFunction
from kani.models import ChatMessage


# ==== completions ====
class BaseCompletion(abc.ABC):
    """Base class for all LM engine completions."""

    @property
    @abc.abstractmethod
    def message(self) -> ChatMessage:
        """The message returned by the LM."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def prompt_tokens(self) -> int | None:
        """How many tokens are in the prompt. Can be None for kani to estimate using tokenizer."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def completion_tokens(self) -> int | None:
        """How many tokens are in the completion. Can be None for kani to estimate using tokenizer."""
        raise NotImplementedError


class Completion(BaseCompletion):
    def __init__(self, message: ChatMessage, prompt_tokens: int | None = None, completion_tokens: int | None = None):
        self._message = message
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    @property
    def message(self):
        return self._message

    @property
    def prompt_tokens(self):
        return self._prompt_tokens

    @property
    def completion_tokens(self):
        return self._completion_tokens


# ==== base engines ====
class BaseEngine(abc.ABC):
    """Base class for all LM engines.

    To add support for a new LM, make a subclass of this and implement the abstract methods below.
    """

    # ==== required interface ====
    max_context_size: int
    """The maximum context size supported by this engine's LM."""

    @abc.abstractmethod
    def message_len(self, message: ChatMessage) -> int:
        """Return the length, in tokens, of the given chat message."""
        raise NotImplementedError

    @abc.abstractmethod
    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> BaseCompletion:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param hyperparams: Any additional parameters to pass to the engine.
        """
        raise NotImplementedError

    # ==== optional interface ====
    token_reserve: int = 0
    """Optional: The number of tokens to reserve for internal engine mechanisms (e.g. if an engine has to set up the
    model's reply with a delimiting token).
    
    Default: 0
    """

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        """Optional: How many tokens are required to build a prompt to expose the given functions to the model.

        Default: If this is not implemented and the user passes in functions, log a warning that the engine does not
        support function calling.
        """
        if functions:
            warnings.warn(
                f"The {type(self).__name__} engine is conversational only and does not support function calling.\n"
                "Developers: If this warning is incorrect, please implement `function_token_reserve()`."
            )
        return 0

    async def stream(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        """
        Optional: Stream a completion from the engine, token-by-token.

        This method's signature is the same as :meth:`.BaseEngine.predict`.

        This method should yield strings as an asynchronous iterable.

        Optionally, this method may also yield a :class:`.BaseCompletion`. If it does, it MUST be the last item
        yielded by this method.

        If an engine does not implement streaming, this method will yield the entire text of the completion in a single
        chunk by default.

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param hyperparams: Any additional parameters to pass to the engine.
        """
        warnings.warn(
            f"This {type(self).__name__} does not implement streaming. This stream will yield the entire completion in"
            " one single chunk."
        )
        completion = await self.predict(messages, functions, **hyperparams)
        yield completion.message.text
        yield completion

    async def close(self):
        """Optional: Clean up any resources the engine might need."""
        pass

    # ==== internal ====
    __ignored_repr_attrs__ = ("token_cache",)

    def __repr__(self):
        """Default: generate a repr based on the instance's __dict__."""
        attrs = ", ".join(
            f"{name}={value!r}"
            for name, value in self.__dict__.items()
            if name not in self.__ignored_repr_attrs__ and not name.startswith("_")
        )
        return f"{type(self).__name__}({attrs})"


# ==== utils ====
class WrapperEngine(BaseEngine):
    """
    A base class for engines that are meant to wrap other engines. By default, this class takes in another engine
    as the first parameter in its constructor and will pass through all non-overriden attributes to the wrapped
    engine.
    """

    def __init__(self, engine: BaseEngine, *args, **kwargs):
        """
        :param engine: The engine to wrap.
        """
        super().__init__(*args, **kwargs)
        self.engine = engine
        """The wrapped engine."""

        # passthrough attrs
        self.max_context_size = engine.max_context_size
        self.token_reserve = engine.token_reserve

    # passthrough methods
    def message_len(self, message: ChatMessage) -> int:
        return self.engine.message_len(message)

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> BaseCompletion:
        return await self.engine.predict(messages, functions, **hyperparams)

    async def stream(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        async for elem in self.engine.stream(messages, functions, **hyperparams):
            yield elem

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        return self.engine.function_token_reserve(functions)

    async def close(self):
        return await self.engine.close()

    def __repr__(self):
        return f"{type(self).__name__}(engine={self.engine!r})"

    # all other attributes are caught by this default passthrough handler
    def __getattr__(self, item):
        return getattr(self.engine, item)
