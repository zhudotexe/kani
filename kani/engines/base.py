import abc

from kani.ai_function import AIFunction
from kani.models import ChatMessage


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


class BaseEngine(abc.ABC):
    """Base class for all LM engines.

    To add support for a new LM, make a subclass of this and implement the abstract methods below.
    """

    max_context_size: int
    """The maximum context size supported by this engine's LM."""

    token_reserve: int = 0
    """The number of tokens to reserve for internal engine mechanisms (e.g. OpenAI's function calling prompt)."""

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

    async def close(self):
        """Clean up any resources the engine might need."""
        pass
