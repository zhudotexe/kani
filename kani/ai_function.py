import asyncio
import inspect
import typing
import warnings
from typing import Annotated

from pydantic import validate_call

from .exceptions import FunctionSpecError
from .json_schema import AIParamSchema, create_json_schema
from .models import ChatRole


class AIFunction:
    """Wrapper around a function to expose to a language model."""

    def __init__(
        self,
        inner,
        after: ChatRole = ChatRole.ASSISTANT,
        name: str | None = None,
        desc: str | None = None,
        auto_retry: bool = True,
        json_schema: dict | None = None,
        auto_truncate: int | None = None,
    ):
        """
        :param inner: The function implementation.
        :param after: Who should speak next after the function call completes (see :ref:`next_actor`). Defaults to the
            model.
        :param name: The name of the function (defaults to the name of the function in source code).
        :param desc: The function's description (defaults to the function's docstring).
        :param auto_retry: Whether the model should retry calling the function if it gets it wrong
            (see :ref:`auto_retry`).
        :param json_schema: A JSON Schema document describing the function's parameters. By default, kani will
            automatically generate one, but this can be helpful for overriding it in any tricky cases.
        :param auto_truncate: If a function response is longer than this many tokens, truncate it until it is at most
            this many tokens and add "..." to the end. By default, no responses will be truncated. This uses a smart
            paragraph-aware truncation algorithm.
        """
        # pydantic's wrapper mangles the async signature so we have to store this here
        self._inner_is_coro = inspect.iscoroutinefunction(inner)
        self.inner = validate_call(inner)
        self.after = after
        self.name = name or inner.__name__
        self.desc = desc or inspect.getdoc(inner)
        self.auto_retry = auto_retry
        self.json_schema = self.create_json_schema() if json_schema is None else json_schema
        self.auto_truncate = auto_truncate

        # wraps() things
        self.__name__ = inner.__name__
        self.__qualname__ = inner.__qualname__
        self.__annotations__ = inner.__annotations__
        self.__module__ = inner.__module__
        self.__doc__ = inner.__doc__

        # validation
        if not self.desc:
            warnings.warn(
                f"The {self.name!r} @ai_function is missing a description. This may lead to request errors or poor"
                ' performance by models. To add a description, add a """docstring""" beneath the signature or use'
                ' @ai_function(desc="...").'
            )

    async def __call__(self, *args, **kwargs):
        if self._inner_is_coro:
            return await self.inner(*args, **kwargs)
        # run synch functions in a thread in order to maintain async safety as best we can
        return await asyncio.to_thread(self.inner, *args, **kwargs)

    def get_params(self) -> list[AIParamSchema]:
        # get list of params
        params = []
        sig = inspect.signature(self.inner)
        type_hints = typing.get_type_hints(self.inner)
        for name, param in sig.parameters.items():
            # ensure param can be supplied thru kwargs
            if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                raise FunctionSpecError(
                    "Positional-only or variadic parameters are not allowed in @ai_function()s."
                    f" ({self.inner.__name__}#{name})"
                )

            # ensure the type is annotated
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise FunctionSpecError(
                    f"All @ai_function() parameters must have a type annotation ({self.inner.__name__}#{name})."
                )

            # ensure type hint matches up
            if name not in type_hints:
                raise RuntimeError(f"The schema generator could not find the type hint ({self.inner.__name__}#{name}).")

            # get aiparam and add it to the list
            ai_param = get_aiparam(annotation)
            params.append(
                AIParamSchema(
                    name=name, t=type_hints[name], default=param.default, aiparam=ai_param, inspect_param=param
                )
            )
        return params

    def create_json_schema(self, include_desc=True) -> dict:
        """
        Create a JSON schema representing this function's parameters as a JSON object.

        :param include_desc: Whether to include the AIFunction's description in the generated JSON schema.
        """
        kwargs = {}
        if include_desc:
            kwargs["desc"] = self.desc
        return create_json_schema(self.get_params(), name=self.name, **kwargs)

    def __repr__(self):
        return (
            f"{type(self).__name__}(name={self.name!r}, desc={self.desc!r}, after={self.after!r},"
            f" auto_retry={self.auto_retry!r}, auto_truncate={self.auto_truncate!r})"
        )


def ai_function(
    func=None,
    *,
    after: ChatRole = ChatRole.ASSISTANT,
    name: str | None = None,
    desc: str | None = None,
    auto_retry: bool = True,
    json_schema: dict | None = None,
    auto_truncate: int | None = None,
):
    """Decorator to mark a method of a Kani to expose to the AI.

    :param after: Who should speak next after the function call completes (see :ref:`next_actor`). Defaults to the
        model.
    :param name: The name of the function (defaults to the name of the function in source code).
    :param desc: The function's description (defaults to the function's docstring).
    :param auto_retry: Whether the model should retry calling the function if it gets it wrong (see :ref:`auto_retry`).
    :param json_schema: A JSON Schema document describing the function's parameters. By default, kani will automatically
        generate one, but this can be helpful for overriding it in any tricky cases.
    :param auto_truncate: If a function response is longer than this many tokens, truncate it until it is at most
        this many tokens and add "..." to the end. By default, no responses will be truncated. This uses a smart
        paragraph-aware truncation algorithm.
    """

    def deco(f):
        f.__ai_function__ = {
            "after": after,
            "name": name or f.__name__,
            "desc": desc or inspect.getdoc(f),
            "auto_retry": auto_retry,
            "json_schema": json_schema,
            "auto_truncate": auto_truncate,
        }
        return f

    if func is not None:
        return deco(func)
    return deco


# ==== AIParam ====
class AIParam:
    """Special tag to annotate types with in order to provide parameter-level metadata to kani."""

    def __init__(self, desc: str, *, title: str = None):
        """
        :param desc: The description of the parameter.
        :param title: If set, set the title of this parameter in generated JSON schema to this; otherwise omit the title
            (as it is already the key of the parameter in the schema).
        """
        self.desc = desc
        self.title = title


def get_aiparam(annotation: type) -> AIParam | None:
    """If the type annotation is an Annotated containing an AIParam, extract and return it."""
    # is it Annotated? if so, get AIParam from annotation
    if typing.get_origin(annotation) is not Annotated:
        return

    for a in typing.get_args(annotation):
        if isinstance(a, AIParam):
            return a
