import pytest

from kani import ChatMessage, ChatRole, MessagePart, ToolCall


class _TestMessagePart(MessagePart):
    def __str__(self):
        return "<TestMessagePart>"


def test_basic():
    msg = ChatMessage(role=ChatRole.USER, content="Hello world")
    assert msg.role == ChatRole.USER
    assert msg.content == "Hello world"
    assert msg.text == "Hello world"
    assert msg.parts == ["Hello world"]
    assert msg.name is None
    assert msg.function_call is None


# v1.0: no more immutability >:c
def test_mutable():
    msg = ChatMessage(role=ChatRole.USER, content="Hello world")
    msg.content = "not allowed"
    assert msg.content == "not allowed"


def test_none_content():
    msg = ChatMessage(role=ChatRole.USER, content=None)
    assert msg.content is None
    assert msg.text is None
    assert msg.parts == []


def test_parts():
    part = _TestMessagePart()
    msg = ChatMessage(role=ChatRole.USER, content=["Hello world", part])
    assert msg.content == ["Hello world", part]
    assert msg.text == "Hello world<TestMessagePart>"
    assert msg.parts == ["Hello world", part]


def test_copy_parts():
    part = _TestMessagePart()
    msg = ChatMessage(role=ChatRole.USER, content=["Hello world", part])

    text_copy = msg.copy_with(text="asdf")
    assert text_copy.content == "asdf"

    part_copy = msg.copy_with(parts=["foo"])
    assert part_copy.content == ["foo"]

    content_copy = msg.copy_with(content="zxcv")
    assert content_copy.content == "zxcv"

    assert msg.content == ["Hello world", part]

    with pytest.raises(ValueError):
        msg.copy_with(text="foo", parts=[])


def test_copy_tools():
    msg = ChatMessage(role=ChatRole.ASSISTANT, content=None)

    bar_call = ToolCall.from_function("bar")
    calls_copy = msg.copy_with(tool_calls=[bar_call])
    assert calls_copy.tool_calls == [bar_call]
    assert calls_copy.function_call == bar_call.function

    func_copy = msg.copy_with(function_call=bar_call.function)
    assert len(func_copy.tool_calls) == 1
    assert func_copy.function_call == bar_call.function

    with pytest.raises(ValueError):
        msg.copy_with(tool_calls=[bar_call], function_call=bar_call.function)


def test_support_multiple_tool_calls():
    tool_call1 = ToolCall.from_function("foo")
    tool_call2 = ToolCall.from_function("bar")

    msg = ChatMessage(role="function", content=None, tool_calls=[tool_call1, tool_call2])

    assert msg.tool_calls == [tool_call1, tool_call2]
