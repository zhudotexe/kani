"""Save -> load should be an identity transformation."""

from hypothesis import HealthCheck, given, settings, strategies as st

from kani import ChatMessage, FunctionCall, Kani, MessagePart
from tests.engine import TestEngine

engine = TestEngine()


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow), deadline=None)
@given(st.data())
async def test_saveload_str(tmp_path, data):
    """Test that basic string content messages are saved."""
    # randomly initialize a kani state
    ai = Kani(
        engine,
        desired_response_tokens=3,
        system_prompt=data.draw(st.text(min_size=0, max_size=1)),
        always_included_messages=[ChatMessage.user(data.draw(st.text(min_size=0, max_size=1)))],
    )
    for _ in range(5):
        query = data.draw(st.text(min_size=0, max_size=5))
        await ai.chat_round_str(query, test_echo=True)

    # save and load
    ai.save(tmp_path / "pytest.json")
    loaded = Kani(engine)
    loaded.load(tmp_path / "pytest.json")

    # assert equality
    assert ai.always_included_messages == loaded.always_included_messages
    assert ai.chat_history == loaded.chat_history


async def test_saveload_tool_calls(tmp_path):
    """Test that tool calls are saved."""
    fewshot = [
        ChatMessage.user("What's the weather in Philadelphia?"),
        ChatMessage.assistant(
            content=None,
            function_call=FunctionCall.with_args("get_weather", location="Philadelphia, PA", unit="fahrenheit"),
        ),
        ChatMessage.function("get_weather", "Weather in Philadelphia, PA: Partly cloudy, 85 degrees fahrenheit."),
        ChatMessage.assistant(
            content=None,
            function_call=FunctionCall.with_args("get_weather", location="Philadelphia, PA", unit="celsius"),
        ),
        ChatMessage.function("get_weather", "Weather in Philadelphia, PA: Partly cloudy, 29 degrees celsius."),
        ChatMessage.assistant("It's currently 85F (29C) and partly cloudy in Philadelphia."),
    ]
    ai = Kani(engine, chat_history=fewshot)

    # save and load
    ai.save(tmp_path / "pytest.json")
    loaded = Kani(engine)
    loaded.load(tmp_path / "pytest.json")

    # assert equality
    assert ai.always_included_messages == loaded.always_included_messages
    assert ai.chat_history == loaded.chat_history


class _TestMessagePart1(MessagePart):
    data: str


class _TestMessagePart2(MessagePart):
    data: str


async def test_saveload_messageparts(tmp_path):
    """Test that message parts are serialized and deserialized into the right classes."""
    apart1 = _TestMessagePart1(data="apart1")
    apart2 = _TestMessagePart2(data="apart2")
    hpart1 = _TestMessagePart1(data="hpart1")
    hpart2 = _TestMessagePart2(data="hpart2")
    # ensure that different instances with the same data are the same
    assert apart1 == _TestMessagePart1(data="apart1")
    # ensure that different classes/data are not
    assert apart1 != _TestMessagePart2(data="apart1")
    assert apart1 != hpart1

    # init kani state
    ai = Kani(
        engine,
        always_included_messages=[
            ChatMessage.user(["astr", apart1]),
            ChatMessage.user([apart2, "astr2"]),
        ],
        chat_history=[
            ChatMessage.user(["hstr", hpart1]),
            ChatMessage.user([hpart2, "hstr2"]),
        ],
    )

    # save and load
    ai.save(tmp_path / "pytest.json")
    loaded = Kani(engine)
    loaded.load(tmp_path / "pytest.json")

    # assert equality
    assert ai.always_included_messages == loaded.always_included_messages
    assert ai.chat_history == loaded.chat_history
