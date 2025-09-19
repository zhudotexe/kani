class KaniException(Exception):
    """Base class for all Kani exceptions/errors."""


class MessageTooLong(KaniException):
    """This chat message will never fit in the context window."""






# ==== function calling ====
class FunctionCallException(KaniException):
    """Base class for exceptions that occur when a model calls an @ai_function."""

    def __init__(self, retry: bool):
        self.retry = retry


class WrappedCallException(FunctionCallException):
    """The @ai_function raised an exception."""

    def __init__(self, retry, original):
        super().__init__(retry)
        self.original = original

    def __str__(self):
        return str(self.original)


class NoSuchFunction(FunctionCallException):
    """The model attempted to call a function that does not exist."""

    def __init__(self, name):
        super().__init__(True)
        self.name = name


# ==== programmer errors ====
class FunctionSpecError(KaniException):
    """This @ai_function spec is invalid."""


class MissingModelDependencies(KaniException):
    """You are trying to use an engine but do not have engine-specific packages installed."""


class PromptError(KaniException):
    """For some reason, the input to this model is invalid."""


# ==== serdes ====
class MissingMessagePartType(KaniException):
    """During loading a saved kani, a message part has a type which is not currently defined in the runtime."""

    def __init__(self, fqn: str, msg: str):
        """
        :param fqn: The fully qualified name of the type that is missing.
        """
        super().__init__(msg)
        self.fqn = fqn
