import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from kani import ChatMessage, ChatRole, Kani
from kani.engines.base import WrapperEngine
from tests.engine import TestEngine, TestStreamingEngine
from tests.utils import flatten_chatmessages

engine = TestEngine()
streaming_engine = TestStreamingEngine()
wrapped_engine = WrapperEngine(engine)
wrapped_streaming_engine = WrapperEngine(streaming_engine)


@pytest.mark.parametrize("eng", [engine, streaming_engine, wrapped_engine, wrapped_streaming_engine])
async def test_chat_round_stream_consume_all(eng):
    ai = Kani(eng, desired_response_tokens=3)
    # 5 tokens, no omitting
    resp = await ai.chat_round_stream("12345")
    assert resp.content == "a"
    assert resp.role == ChatRole.ASSISTANT
    assert len(ai.chat_history) == 2
    prompt = await ai.get_prompt()
    assert len(prompt) == 2  # 12345a
    assert flatten_chatmessages(prompt) == "12345a"


@pytest.mark.parametrize("eng", [streaming_engine, wrapped_streaming_engine])
async def test_chat_round_stream(eng):
    ai = Kani(eng, desired_response_tokens=3)
    stream = ai.chat_round_stream("12345")
    async for token in stream:
        assert token == "a"
    resp = await stream.message()
    assert resp.content == "a"

    ai = Kani(eng, desired_response_tokens=3)
    stream = ai.chat_round_stream("aaa", test_echo=True)
    async for token in stream:
        assert token == "a"
    resp = await stream.message()
    assert resp.content == "aaa"


@settings(suppress_health_check=(HealthCheck.too_slow,), deadline=None)
@given(st.data())
@pytest.mark.parametrize("eng", [engine, streaming_engine, wrapped_engine, wrapped_streaming_engine])
async def test_spam_stream(eng, data):
    # spam the kani with a bunch of random prompts
    # and make sure it never breaks
    ai = Kani(
        eng,
        desired_response_tokens=3,
        system_prompt=data.draw(st.text(min_size=0, max_size=1)),
        always_included_messages=[ChatMessage.user(data.draw(st.text(min_size=0, max_size=1)))],
    )
    queries = data.draw(st.lists(st.text(min_size=0, max_size=5)))
    for query in queries:
        query = query.strip()
        resp = await ai.chat_round_stream(query, test_echo=True)
        assert resp.content == query

        prompt = await ai.get_prompt()
        assert sum(ai.message_token_len(m) for m in prompt) <= ai.max_context_size
