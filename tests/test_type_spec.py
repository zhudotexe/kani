import enum
from typing import Annotated, List

from chatterbox import AIParam


# setup
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


async def example_primitives(
    a: str,
    b: float,
    c: Annotated[str, AIParam(desc="I am C")],
    d: Annotated[int, "I am not an AIParam"],
):
    """description!"""
    pass


async def example_collections(
    a: list[str],
    b: dict[str, int],
    c: Annotated[list[str], AIParam(desc="I am C")],
    d: Annotated[list[int], "I am not an AIParam"],
    e: List[str],
):
    """collections!"""
    pass


async def example_enums(
    a: EnumS,
    b: EnumI,
    c: EnumI2,
    d: Annotated[EnumS, AIParam(desc="I am D")],
    e: Annotated[EnumI, "I am not an AIParam"],
):
    """enums!"""
    pass


# tests
