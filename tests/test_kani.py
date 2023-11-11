from hypothesis import HealthCheck, given, settings, strategies as st

from kani import ChatMessage, ChatRole, Kani
from tests.engine import TestEngine
from tests.utils import flatten_chatmessages

engine = TestEngine()


async def test_chat_round():
    ai = Kani(engine, desired_response_tokens=3)
    # 5 tokens, no omitting
    resp = await ai.chat_round("12345")
    assert resp.content == "a"
    assert resp.role == ChatRole.ASSISTANT
    assert len(ai.chat_history) == 2
    prompt = await ai.get_prompt()
    assert len(prompt) == 2  # 12345a
    assert flatten_chatmessages(prompt) == "12345a"

    # 5 more, first message should be omitted
    resp = await ai.chat_round_str("67890")
    assert resp == "a"
    assert len(ai.chat_history) == 4
    assert flatten_chatmessages(ai.chat_history) == "12345a67890a"
    prompt = await ai.get_prompt()
    assert len(prompt) == 3  # 12345a
    assert flatten_chatmessages(prompt) == "a67890a"


async def test_always_include():
    # always include 2 tokens, reserve 3 for response
    ai = Kani(engine, desired_response_tokens=3, system_prompt="1", always_included_messages=[ChatMessage.user("2")])
    assert len(ai.always_included_messages) == 2
    assert sum(ai.message_token_len(m) for m in ai.always_included_messages) == 2

    # messages are only included if <= 5 tokens
    resp = await ai.chat_round_str("12345")
    assert resp == "a"
    assert len(ai.chat_history) == 2  # always include are not included in chat history
    assert flatten_chatmessages(ai.chat_history) == "12345a"
    prompt = await ai.get_prompt()
    assert len(prompt) == 3  # 12a (12345 gets dropped)
    assert flatten_chatmessages(prompt) == "12a"


@settings(suppress_health_check=(HealthCheck.too_slow,), deadline=None)
@given(st.data())
async def test_spam(data):
    # spam the kani with a bunch of random prompts
    # and make sure it never breaks
    ai = Kani(
        engine,
        desired_response_tokens=3,
        system_prompt=data.draw(st.text(min_size=0, max_size=1)),
        always_included_messages=[ChatMessage.user(data.draw(st.text(min_size=0, max_size=1)))],
    )
    queries = data.draw(st.lists(st.text(min_size=0, max_size=5)))
    for query in queries:
        resp = await ai.chat_round_str(query, test_echo=True)
        assert resp == query

        prompt = await ai.get_prompt()
        assert sum(ai.message_token_len(m) for m in prompt) <= ai.max_context_size
