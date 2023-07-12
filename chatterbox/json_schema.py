import typing
from enum import Enum
from typing import Type, Collection

from .aiparam import AIParam
from .exceptions import FunctionSpecError


class AIParamSchema:
    """Used to annotate parameters of AIFunctions in order to make generating their schema nicer."""

    def __init__(self, name: str, t: type, required: bool, aiparam: AIParam | None = None):
        self.name = name
        self.type = t
        self.required = required
        self.aiparam = aiparam

    @property
    def origin_type(self):
        """If the type takes parameters (e.g. list[...]), the base type (i.e. list). Otherwise same as the type."""
        return typing.get_origin(self.type) or self.type


class JSONSchemaGenerator:
    def __init__(self, params: list[AIParamSchema]):
        self.params = params
        self.seen = set()  # types can be recursive; if so ask the user to manually supply their schema

    def generate(self) -> dict:
        required_params = [s.name for s in self.params if s.required]
        props = {}
        for param in self.params:
            props[param.name] = type_to_json_schema(param.type)
            if param.aiparam and param.aiparam.desc:
                props[param.name]["description"] = param.aiparam.desc
        return {
            "type": "object",
            "properties": props,
            "required": required_params,
        }


TYPE_NAMES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _collection_to_json_schema(t: Type[Collection]):
    pass


def _enum_to_json_schema(t: Type[Enum]):
    pass


def type_to_json_schema(t: type) -> dict:
    """Translate a resolved type (i.e. after `typing.get_type_hints()`) to a partial JSON schema dict."""
    origin_type = typing.get_origin(t) or t
    if issubclass(origin_type, (list, set)):
        return _collection_to_json_schema(t)
    elif isinstance(origin_type, Enum):
        return _enum_to_json_schema(t)
    # elif isinstance(origin_type,)  # todo dataclasses
    # {
    #                 "location": {
    #                     "type": "string",
    #                     "description": "The city and state, e.g. San Francisco, CA",
    #                 },
    #                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    #             }
    # primitive type
    if t not in TYPE_NAMES:
        raise FunctionSpecError(f"{t!r} is not a valid parameter type for an @ai_function.")
    # noinspection PyTypeChecker
    return {
        "type": TYPE_NAMES[t],
    }
