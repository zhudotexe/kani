import abc

from .models import ChatMessage


class HasMessage(abc.ABC):
    message: ChatMessage


class FunctionCallResult(HasMessage):
    """A model requested a function call, and the kani runtime resolved it."""

    def __init__(self, is_model_turn: bool, message: ChatMessage):
        """
        :param is_model_turn: True if the model should immediately react; False if the user speaks next.
        :param message: The message containing the result of the function call, to add to the chat history.
        """
        self.is_model_turn = is_model_turn
        self.message = message


class ExceptionHandleResult(HasMessage):
    """A function call raised an exception, and the kani runtime has prompted the model with exception information."""

    def __init__(self, should_retry: bool, message: ChatMessage):
        """
        :param should_retry: Whether the model should be allowed to retry the call that caused this exception.
        :param message: The message containing details about the exception and/or instructions to retry, to add to the
            chat history.
        """
        self.should_retry = should_retry
        self.message = message
