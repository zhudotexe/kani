import functools
import sys
import warnings
from typing import Callable, ParamSpec, TypeVar


def warn_in_userspace(message, stacklevel=2, **kwargs):
    """
    Issue a warning at the given stacklevel (Python 3.12 or lower) or at the first frame not in the library (3.12+).
    """
    if sys.version_info >= (3, 12):
        _warnings_kwargs = {"skip_file_prefixes": ("kani",)}
    else:
        _warnings_kwargs = {"stacklevel": stacklevel + 1}  # +1 to account for the frame in this function
    warnings.warn(message=message, **_warnings_kwargs, **kwargs)


rT = TypeVar("rT")  # return type
pT = ParamSpec("pT")  # parameters type


def deprecated(
    msg=None, *, category=DeprecationWarning, stacklevel=2
) -> Callable[[Callable[pT, rT]], Callable[pT, rT]]:
    """
    Use this decorator to mark functions as deprecated.
    Every time the decorated function runs, it will emit
    a "deprecation" warning.
    """

    def decorator(func: Callable[pT, rT]) -> Callable[pT, rT]:
        nonlocal msg
        if msg is None:
            msg = f"Function is deprecated: {func.__name__}"
        func.__deprecated__ = msg  # py3.13 static type checking compat

        @functools.wraps(func)
        def new_func(*args: pT.args, **kwargs: pT.kwargs):
            warnings.warn(msg, category=category, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return new_func

    return decorator
