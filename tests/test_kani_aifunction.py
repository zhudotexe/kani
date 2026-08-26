import json

import pytest
from pydantic import BaseModel

from kani import AIFunction, ChatMessage, ChatRole, Kani, ai_function
from kani.exceptions import FunctionSpecError, NoSuchFunction, WrappedCallException
from kani.models import FunctionCall
from kani.parts.text import TextPart
from tests.engine import EngineForTests

engine = EngineForTests()


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


# ==== do_function_call: return-value coercion ====
class _Weather(BaseModel):
    temp: int
    conditions: str


class ReturnTypesKani(Kani):
    @ai_function()
    def ret_str(self):
        """Returns a plain string."""
        return "plain string"

    @ai_function()
    def ret_dict(self):
        """Returns a dict."""
        return {"key": "value", "n": 1}

    @ai_function()
    def ret_pydantic(self):
        """Returns a pydantic model."""
        return _Weather(temp=72, conditions="sunny")

    @ai_function()
    def ret_chatmessage(self):
        """Returns a ChatMessage directly."""
        return ChatMessage.function("original_name", "custom content", tool_call_id="will_be_overwritten")

    @ai_function()
    def ret_messageparts(self):
        """Returns a list containing a MessagePart."""
        return [TextPart(content="hello from parts")]

    @ai_function(after=ChatRole.USER)
    def ret_after_user(self):
        """Yields the turn back to the user after running."""
        return "done"

    @ai_function(auto_truncate=10)
    def ret_long(self):
        """Returns a string longer than the auto_truncate limit."""
        return "word " * 30

    @ai_function()
    def ret_unserializable(self):
        """Returns a dict that is not JSON-serializable."""
        return {"obj": object()}

    @ai_function(auto_truncate=10)
    def ret_untruncatable(self):
        """Returns a long string with no natural break points."""
        return "x" * 50

    @ai_function()
    def raiser(self):
        """Always raises."""
        raise ValueError("boom")


async def _call(fn_name: str, tool_call_id: str = "tc1"):
    ai = ReturnTypesKani(engine)
    return await ai.do_function_call(FunctionCall.with_args(fn_name), tool_call_id=tool_call_id)


async def test_do_function_call_str_result():
    result = await _call("ret_str")
    assert result.message.role == ChatRole.FUNCTION
    assert result.message.name == "ret_str"
    assert result.message.text == "plain string"
    assert result.message.tool_call_id == "tc1"
    assert result.is_model_turn is True  # default after=ASSISTANT


async def test_do_function_call_dict_result_is_json_serialized():
    result = await _call("ret_dict")
    assert json.loads(result.message.text) == {"key": "value", "n": 1}


async def test_do_function_call_pydantic_result_is_json_serialized():
    result = await _call("ret_pydantic")
    assert json.loads(result.message.text) == {"temp": 72, "conditions": "sunny"}


async def test_do_function_call_chatmessage_result_passthrough_with_id():
    result = await _call("ret_chatmessage")
    assert result.message.text == "custom content"
    # do_function_call overwrites the tool_call_id so the FUNCTION message is bound to the request
    assert result.message.tool_call_id == "tc1"


async def test_do_function_call_messagepart_result_preserved():
    result = await _call("ret_messageparts")
    assert isinstance(result.message.parts[0], TextPart)
    assert result.message.text == "hello from parts"


async def test_do_function_call_after_user_is_not_model_turn():
    result = await _call("ret_after_user")
    assert result.is_model_turn is False


async def test_do_function_call_auto_truncates_long_result():
    result = await _call("ret_long")
    assert len(result.message.text) <= 10
    assert result.message.text.endswith("...")


async def test_do_function_call_non_serializable_dict_falls_back_to_str():
    result = await _call("ret_unserializable")
    # json.dumps fails on the object(), so it falls back to str(result)
    assert "object" in result.message.text


async def test_do_function_call_auto_truncate_hard_chops_unbreakable_text():
    result = await _call("ret_untruncatable")
    # no divider to break on, so it hard-chops to max_len characters
    assert result.message.text == "x" * 7 + "..."


async def test_auto_truncate_returns_empty_message_unchanged():
    ai = ReturnTypesKani(engine)
    msg = ChatMessage.function("f", "")
    assert ai._auto_truncate_message(msg, max_len=5).text == ""


async def test_do_function_call_wraps_raised_exception():
    with pytest.raises(WrappedCallException):
        await _call("raiser")


async def test_do_function_call_unknown_function_raises():
    ai = ReturnTypesKani(engine)
    with pytest.raises(NoSuchFunction):
        await ai.do_function_call(FunctionCall.with_args("does_not_exist"))


# ==== AIFunction signature validation ====
def test_ai_function_requires_type_annotations():
    def bad(x):
        """Missing an annotation on x."""

    with pytest.raises(FunctionSpecError, match="type annotation"):
        AIFunction(bad)


def test_ai_function_rejects_positional_only_params():
    def bad(x: int, /):
        """Positional-only params are not allowed."""

    with pytest.raises(FunctionSpecError, match="Positional-only or variadic"):
        AIFunction(bad)


def test_ai_function_rejects_variadic_params():
    def bad(*args: int):
        """Variadic params are not allowed."""

    with pytest.raises(FunctionSpecError, match="Positional-only or variadic"):
        AIFunction(bad)


def test_ai_function_warns_without_description():
    def nodoc(x: int):
        pass

    with pytest.warns(UserWarning, match="missing a description"):
        AIFunction(nodoc)
