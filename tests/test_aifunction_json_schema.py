import enum
from typing import Annotated, List

from kani import AIParam, ai_function
from kani.ai_function import AIFunction
from .utils import dict_at_least


# ==== setup ====
class EnumS(enum.Enum):
    APPLE = "apple"
    BANANA = "banana"
    COCONUT = "coconut"


class EnumI(enum.Enum):
    ONE = 1
    TWO = 2
    THREE = 3


class EnumI2(enum.IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3


class BadEnum(enum.Enum):
    FOO = 1
    BAR = "two"


# noinspection PyUnusedLocal
async def example_primitives(
    a: str,
    b: float,
    c: bool,
    d: int,
    e: type(None),  # https://github.com/pydantic/pydantic/issues/11073
    aa: Annotated[str, AIParam(desc="I am AA")],
    dd: Annotated[int, "I am not an AIParam"] = 2,
):
    """description!"""
    pass


# noinspection PyUnusedLocal
async def example_collections(
    a: list[str],
    b: dict[str, int],
    c: Annotated[list[str], AIParam(desc="I am C")],
    d: Annotated[list[int], "I am not an AIParam"],
    e: List[str],
):
    """collections!"""
    pass


# noinspection PyUnusedLocal
async def example_enums(
    a: EnumS,
    b: EnumI,
    c: EnumI2,
    d: Annotated[EnumS, AIParam(desc="I am D")],
    e: Annotated[EnumI, "I am not an AIParam"],
):
    """enums!"""
    pass


def create_ai_function(f):
    decorated = ai_function(f)
    return AIFunction(decorated, **decorated.__ai_function__)


# ==== tests ====
def test_schema_primitives():
    f = create_ai_function(example_primitives)
    assert isinstance(f, AIFunction)
    assert dict_at_least(
        f.json_schema,
        {
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"},
                "c": {"type": "boolean"},
                "d": {"type": "integer"},
                "e": {"type": "null"},
                "aa": {"description": "I am AA", "type": "string"},
                "dd": {"default": 2, "type": "integer"},
            },
            "required": ["a", "b", "c", "d", "e", "aa"],
            "type": "object",
        },
    )


def test_schema_collections():
    f = create_ai_function(example_collections)
    assert isinstance(f, AIFunction)
    assert dict_at_least(
        f.json_schema,
        {
            "properties": {
                "a": {"items": {"type": "string"}, "type": "array"},
                "b": {"additionalProperties": {"type": "integer"}, "type": "object"},
                "c": {"description": "I am C", "items": {"type": "string"}, "type": "array"},
                "d": {"items": {"type": "integer"}, "type": "array"},
                "e": {"items": {"type": "string"}, "type": "array"},
            },
            "required": ["a", "b", "c", "d", "e"],
            "type": "object",
        },
    )


def test_schema_enums():
    f = create_ai_function(example_enums)
    assert isinstance(f, AIFunction)
    assert dict_at_least(
        f.json_schema,
        {
            "properties": {
                "a": {"enum": ["apple", "banana", "coconut"], "type": "string"},
                "b": {"enum": [1, 2, 3], "type": "integer"},
                "c": {"enum": [1, 2, 3], "type": "integer"},
                "d": {"enum": ["apple", "banana", "coconut"], "type": "string", "description": "I am D"},
                "e": {"enum": [1, 2, 3], "type": "integer"},
            },
            "required": ["a", "b", "c", "d", "e"],
            "type": "object",
        },
    )
