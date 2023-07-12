class ChatterboxException(Exception):
    """Base class for all Chatterbox exceptions/errors."""

    pass


# ==== HTTP ====
class HTTPException(ChatterboxException):
    pass


class HTTPTimeout(HTTPException):
    pass


class HTTPStatusException(HTTPException):
    def __init__(self, status_code: int, msg: str):
        super().__init__(msg)
        self.status_code = status_code


# ==== function calling ====
class FunctionCallException(ChatterboxException):
    def __init__(self, retry: bool):
        self.retry = retry


class WrappedCallException(FunctionCallException):
    def __init__(self, retry, original):
        super().__init__(retry)
        self.original = original

    def __str__(self):
        return str(self.original)


class NoSuchFunction(FunctionCallException):
    def __init__(self, name):
        super().__init__(True)
        self.name = name


# ==== programmer errors ====
class FunctionSpecError(ChatterboxException):
    pass
