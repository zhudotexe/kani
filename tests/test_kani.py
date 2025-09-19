import math

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from kani import ChatMessage, ChatRole, Kani
from kani.exceptions import MessageTooLong, PromptTooLong
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


@given(st.data())
async def test_get_prompt_optimal(data):
    # make sure get_prompt always returns the maximum number of tokens it can
    ctx_size = data.draw(st.integers(min_value=4, max_value=100))
    print("ctx size:", ctx_size)
    engine = TestEngine(max_context_size=ctx_size)
    ai = Kani(
        engine,
        desired_response_tokens=1,
        system_prompt=data.draw(st.text(min_size=0, max_size=1)),
        always_included_messages=[ChatMessage.user(data.draw(st.text(min_size=0, max_size=1)))],
        chat_history=[ChatMessage.user("a")]
        * (engine.max_context_size + data.draw(st.integers(min_value=1, max_value=100))),
    )
    print("optimal n_iters:", math.ceil(math.log2(len(ai.chat_history))))
    prompt = await ai.get_prompt()
    prompt_len = sum(ai.message_token_len(m) for m in prompt)
    print("prompt len:", prompt_len)
    print()
    assert prompt_len == (ai.max_context_size - ai.desired_response_tokens)


async def test_message_too_long():
    ai = Kani(engine, desired_response_tokens=9, chat_history=[ChatMessage.user("aaa")])
    with pytest.raises(MessageTooLong):
        await ai.get_prompt()
    with pytest.raises(MessageTooLong):
        await ai.chat_round_str("aaa")


async def test_prompt_too_long():
    ai = Kani(engine, desired_response_tokens=9, system_prompt="aaa", chat_history=[ChatMessage.user("a")])
    with pytest.raises(PromptTooLong):
        await ai.get_prompt()
    with pytest.raises(PromptTooLong):
        await ai.chat_round_str("a")


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
        assert sum(ai.message_token_len(m) for m in prompt) <= (ai.max_context_size - ai.desired_response_tokens)
