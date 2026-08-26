"""Tests for the TokenCached mixin used by engines to memoize token counts."""

from kani import AIFunction, ChatMessage, ToolCall
from kani.engines.base import BaseEngine
from kani.engines.mixins import TokenCached


class CachedEngine(TokenCached, BaseEngine):
    max_context_size = 4096

    def prompt_len(self, messages, functions=None, **kwargs):
        return sum(len(m.text or "") for m in messages)

    async def predict(self, messages, functions=None, **hyperparams):
        raise NotImplementedError


def _weather_fn():
    def get_weather(location: str):
        """Get the weather."""

    return AIFunction(get_weather)


# ==== message-level cache ====
def test_message_cache_roundtrip():
    engine = CachedEngine()
    msg = ChatMessage.user("hello")
    assert engine.get_cached_message_len(msg) is None
    engine.set_cached_message_len(msg, 5)
    assert engine.get_cached_message_len(msg) == 5


def test_message_cache_key_depends_on_role_and_content():
    engine = CachedEngine()
    engine.set_cached_message_len(ChatMessage.user("hello"), 5)
    # same text, different role -> different key -> cache miss
    assert engine.get_cached_message_len(ChatMessage.assistant("hello")) is None
    # same role and text -> cache hit
    assert engine.get_cached_message_len(ChatMessage.user("hello")) == 5


def test_message_cache_key_includes_tool_calls():
    engine = CachedEngine()
    with_tools = ChatMessage.assistant(None, tool_calls=[ToolCall.from_function("f", x=1)])
    engine.set_cached_message_len(with_tools, 9)
    # a different tool call (different args) should not collide
    other = ChatMessage.assistant(None, tool_calls=[ToolCall.from_function("f", x=2)])
    assert engine.get_cached_message_len(other) is None
    assert engine.get_cached_message_len(with_tools) == 9


# ==== prompt-level cache ====
def test_prompt_cache_roundtrip_without_functions():
    engine = CachedEngine()
    messages = [ChatMessage.user("hi"), ChatMessage.assistant("yo")]
    assert engine.get_cached_prompt_len(messages, None) is None
    engine.set_cached_prompt_len(messages, None, 4)
    assert engine.get_cached_prompt_len(messages, None) == 4


def test_prompt_cache_distinguishes_functions():
    engine = CachedEngine()
    messages = [ChatMessage.user("hi")]
    engine.set_cached_prompt_len(messages, None, 2)
    # same messages but with functions exposed -> different key
    assert engine.get_cached_prompt_len(messages, [_weather_fn()]) is None
    assert engine.get_cached_prompt_len(messages, None) == 2


def test_prompt_cache_disabled_when_kwargs_present():
    # kwargs make the key unrepresentable, so caching is a no-op (not an error)
    engine = CachedEngine()
    messages = [ChatMessage.user("hi")]
    engine.set_cached_prompt_len(messages, None, 2, temperature=0.5)
    assert engine.get_cached_prompt_len(messages, None, temperature=0.5) is None
