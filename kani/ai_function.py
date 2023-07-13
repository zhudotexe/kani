import inspect
import typing
from typing import Annotated

from .exceptions import FunctionSpecError
from .json_schema import AIParamSchema, create_json_schema
from .models import ChatRole


class AIFunction:
    def __init__(self, inner, after: ChatRole, name: str, desc: str, auto_retry: bool, json_schema: dict = None):
        self.inner = inner
        self.after = after
        self.name = name
        self.desc = desc
        self.auto_retry = auto_retry
        self.json_schema = self.create_json_schema() if json_schema is None else json_schema

        # wraps() things
        self.__name__ = inner.__name__
        self.__qualname__ = inner.__qualname__
        self.__annotations__ = inner.__annotations__
        self.__module__ = inner.__module__
        self.__doc__ = inner.__doc__

    async def __call__(self, *args, **kwargs):
        result = self.inner(*args, **kwargs)
        if inspect.iscoroutine(result):
            return await result
        return result

    def create_json_schema(self) -> dict:
        """create a JSON schema representing this function's parameters as a JSON object."""
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
            params.append(AIParamSchema(name=name, t=type_hints[name], default=param.default, aiparam=ai_param))
        # create a schema generator and generate
        return create_json_schema(params)


def ai_function(
    func=None,
    *,
    after: ChatRole = ChatRole.ASSISTANT,
    name: str | None = None,
    desc: str | None = None,
    auto_retry: bool = True,
    json_schema: dict | None = None,
):
    """Decorator to mark a method of a KaniWithFunctions to expose to the AI.

    **Type Annotations**
    Kani will automatically generate the function schema based off of the function's type annotations.
    The allowed types are:

    - Python primitive types (``None``, :class:`bool`, :class:`str`, :class:`int`, :class:`float`)
    - an enum (subclass of ``enum.Enum``)
    - a list or dict of the above types (e.g. ``list[str]``, ``dict[str, int]``, ``list[SomeEnum]``)

    When the AI calls into the function, Kani guarantees that the passed parameters are of the annotated type.

    **Name & Descriptions**
    If not specified, the function description will be taken from its docstring, and name from the source.
    To specify descriptions of or override the name of a parameter, provide an :class:`AIParam` annotation using an
    Annotated type annotation.

    **Next Actor**
    After a function call returns, Kani will hand control back to the LM to generate a response. If instead
    control should be given to the human (i.e. return from the chat round), set ``after=ChatRole.USER``.

    **Example**
    Here is an example of how you might implement a function to get weather::

        class Unit(enum.Enum):
            FAHRENHEIT = "fahrenheit"
            CELSIUS = "celsius"

        @ai_function()
        async def get_weather(
            location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
            unit: Unit,
        ):
            \"""Get the current weather in a given location.\"""
            ...

    :param after: After completing the function call, who should speak next.
    :param name: The name of the function (defaults to f.__name__)
    :param desc: The desc of the function (defaults to docstring)
    :param auto_retry: Whether the model should retry calling the function if it gets it wrong.
    :param json_schema: If not using autogeneration, the JSON Schema to provide the model.
    """

    def deco(f):
        f.__ai_function__ = {
            "after": after,
            "name": name or f.__name__,
            "desc": desc or inspect.getdoc(f),
            "auto_retry": auto_retry,
            "json_schema": json_schema,
        }
        return f

    if func is not None:
        return deco(func)
    return deco


# ==== AIParam ====
class AIParam:
    """Special tag to annotate types with in order to provide parameter-level metadata to kani."""

    def __init__(self, desc: str):
        self.desc = desc


def get_aiparam(annotation: type) -> AIParam | None:
    """If the type annotation is an Annotated containing an AIParam, extract and return it."""
    # is it Annotated? if so, get AIParam from annotation
    if typing.get_origin(annotation) is not Annotated:
        return

    for a in typing.get_args(annotation):
        if isinstance(a, AIParam):
            return a
