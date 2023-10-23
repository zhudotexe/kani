"""Save -> load should be an identity transformation."""
import random
import string

from kani import ChatMessage, Kani, MessagePart
from tests.engine import TestEngine

engine = TestEngine()


async def test_saveload_str(tmp_path):
    """Test that basic string content messages are saved."""
    # randomly initialize a kani state
    ai = Kani(engine, desired_response_tokens=3, system_prompt="1", always_included_messages=[ChatMessage.user("2")])
    for _ in range(5):
        query_len = random.randint(0, 5)
        query = "".join(random.choice(string.ascii_letters) for _ in range(query_len))
        await ai.chat_round_str(query, test_echo=True)

    # save and load
    ai.save(tmp_path / "pytest.json")
    loaded = Kani(engine)
    loaded.load(tmp_path / "pytest.json")

    # assert equality
    assert ai.always_included_messages == loaded.always_included_messages
    assert ai.chat_history == loaded.chat_history


class TestMessagePart1(MessagePart):
    data: str


class TestMessagePart2(MessagePart):
    data: str


async def test_saveload_messageparts(tmp_path):
    """Test that message parts are serialized and deserialized into the right classes."""
    apart1 = TestMessagePart1(data="apart1")
    apart2 = TestMessagePart2(data="apart2")
    hpart1 = TestMessagePart1(data="hpart1")
    hpart2 = TestMessagePart2(data="hpart2")
    # ensure that different instances with the same data are the same
    assert apart1 == TestMessagePart1(data="apart1")
    # ensure that different classes/data are not
    assert apart1 != TestMessagePart2(data="apart1")
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
