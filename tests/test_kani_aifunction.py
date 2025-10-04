from kani import Kani, ai_function
from tests.engine import TestEngine

engine = TestEngine()


# define a dummy kani class to test
class MyKani(Kani):
    @ai_function()
    def foo1(self):
        """I am foo1"""
        raise RuntimeError("I shouldn't be executed in discovery")

    @ai_function(name="bar2", desc="I am actually bar2")
    def foo2(self):
        """This is not my real docstring"""
        raise RuntimeError("I shouldn't be executed in discovery")

    @ai_function()
    def foo3(self, i: int, s: str):
        """I am foo3 and I have parameters"""
        raise RuntimeError("I shouldn't be executed in discovery")

    def not_an_aifunction(self):
        raise RuntimeError("oh no I shouldn't be executed")

    @property
    def still_not_an_aifunction(self):
        raise RuntimeError("oh no I shouldn't be executed")


async def test_no_functions():
    ai = Kani(engine)
    assert not ai.functions


async def test_function_discovery():
    ai = MyKani(engine)
    assert ai.functions
    assert "foo1" in ai.functions
    assert "foo2" not in ai.functions
    assert "bar2" in ai.functions
    assert "foo3" in ai.functions
    assert "not_an_aifunction" not in ai.functions
    assert "still_not_an_aifunction" not in ai.functions


async def test_function_desc_from_docstring_or_param():
    ai = MyKani(engine)
    assert ai.functions["foo1"].desc == "I am foo1"
    assert ai.functions["bar2"].desc == "I am actually bar2"
