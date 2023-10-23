import pydantic
import pytest

from kani import ChatMessage, ChatRole, MessagePart


class TestMessagePart(MessagePart):
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


def test_immutable():
    msg = ChatMessage(role=ChatRole.USER, content="Hello world")
    with pytest.raises(pydantic.ValidationError):
        msg.content = "not allowed"
    assert msg.content == "Hello world"


def test_none_content():
    msg = ChatMessage(role=ChatRole.USER, content=None)
    assert msg.content is None
    assert msg.text is None
    assert msg.parts == []


def test_parts():
    part = TestMessagePart()
    msg = ChatMessage(role=ChatRole.USER, content=["Hello world", part])
    assert msg.content == ("Hello world", part)
    assert msg.text == "Hello world<TestMessagePart>"
    assert msg.parts == ["Hello world", part]


def test_copy():
    part = TestMessagePart()
    msg = ChatMessage(role=ChatRole.USER, content=["Hello world", part])

    text_copy = msg.copy_with(text="asdf")
    assert text_copy.content == "asdf"

    part_copy = msg.copy_with(parts=["foo"])
    assert part_copy.content == ("foo",)

    content_copy = msg.copy_with(content="zxcv")
    assert content_copy.content == "zxcv"

    assert msg.content == ("Hello world", part)

    with pytest.raises(ValueError):
        msg.copy_with(text="foo", parts=[])
