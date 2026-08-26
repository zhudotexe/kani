"""Tests for the ``full_round`` / ``_full_round`` function-calling orchestration loop.

These use a scripted engine that emits a predetermined sequence of assistant messages (some with tool calls) so the
loop's multi-round, parallel, retry, and turn-yielding behavior can be exercised without a real model.
"""

from typing import AsyncIterable

import pytest

from kani import ChatMessage, ChatRole, Kani, ai_function
from kani.engines.base import BaseCompletion, BaseEngine, Completion
from kani.exceptions import NoSuchFunction, WrappedCallException
from kani.internal import ExceptionHandleResult
from kani.models import FunctionCall, ToolCall


class ScriptedEngine(BaseEngine):
    """Returns a predetermined assistant message on each predict()/stream() call.

    Records whether functions were exposed on each call so tests can assert when the loop disables function calling.
    """

    max_context_size = 4096

    def __init__(self, turns: list[ChatMessage]):
        self.turns = list(turns)
        self.functions_seen: list[bool] = []  # one entry per predict/stream call: were functions passed?

    def prompt_len(self, messages, functions=None, **kwargs):
        return sum(len(m.text or "") for m in messages)

    def _next(self, functions) -> ChatMessage:
        self.functions_seen.append(functions is not None)
        return self.turns.pop(0)

    async def predict(self, messages, functions=None, **hyperparams) -> Completion:
        return Completion(self._next(functions))

    async def stream(self, messages, functions=None, **hyperparams) -> AsyncIterable[str | BaseCompletion]:
        msg = self._next(functions)
        if msg.text is not None:
            yield msg.text
        yield Completion(msg)

    @property
    def predict_count(self) -> int:
        return len(self.functions_seen)


class OrchestrationKani(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []  # records each function that actually executed

    @ai_function()
    def get_time(self):
        """Get the current time."""
        self.calls.append("get_time")
        return "12:00 noon"

    @ai_function()
    def add(self, a: int, b: int):
        """Add two numbers."""
        self.calls.append(("add", a, b))
        return a + b

    @ai_function()
    def boom(self):
        """Always raises; retryable by default."""
        raise ValueError("boom")

    @ai_function(auto_retry=False)
    def boom_noretry(self):
        """Always raises; not retryable."""
        raise ValueError("no retry boom")

    @ai_function(after=ChatRole.USER)
    def notify_user(self):
        """Notify the user; yields the turn back to them."""
        self.calls.append("notify_user")
        return "notified"


def _assistant_call(name: str, **kwargs) -> ChatMessage:
    return ChatMessage.assistant(content=None, tool_calls=[ToolCall.from_function(name, **kwargs)])


async def _collect(ai: Kani, query) -> list[ChatMessage]:
    return [msg async for msg in ai.full_round(query)]


# ==== basic multi-round ====
async def test_full_round_single_tool_call():
    engine = ScriptedEngine([_assistant_call("get_time"), ChatMessage.assistant("It is noon.")])
    ai = OrchestrationKani(engine)
    messages = await _collect(ai, "What time is it?")

    assert [m.role for m in messages] == [ChatRole.ASSISTANT, ChatRole.FUNCTION, ChatRole.ASSISTANT]
    assert messages[1].text == "12:00 noon"
    assert messages[1].is_tool_call_error is False
    assert messages[2].text == "It is noon."
    assert ai.calls == ["get_time"]
    assert engine.predict_count == 2
    # the user query, the tool request, the result, and the final answer are all in history
    assert len(ai.chat_history) == 4


async def test_full_round_parallel_tool_calls():
    both = ChatMessage.assistant(
        content=None,
        tool_calls=[ToolCall.from_function("add", a=1, b=2), ToolCall.from_function("add", a=3, b=4)],
    )
    engine = ScriptedEngine([both, ChatMessage.assistant("3 and 7.")])
    ai = OrchestrationKani(engine)
    messages = await _collect(ai, "Add some numbers.")

    # results are gathered in request order, even though the calls execute concurrently
    function_results = [m for m in messages if m.role == ChatRole.FUNCTION]
    assert [m.text for m in function_results] == ["3", "7"]
    # both ran; execution order is not guaranteed since they run in parallel
    assert set(ai.calls) == {("add", 1, 2), ("add", 3, 4)}


async def test_full_round_after_user_ends_round_without_followup():
    engine = ScriptedEngine([_assistant_call("notify_user")])
    ai = OrchestrationKani(engine)
    messages = await _collect(ai, "Tell the user something.")

    # after=USER means the model does not get another turn: only the tool request + result are produced
    assert [m.role for m in messages] == [ChatRole.ASSISTANT, ChatRole.FUNCTION]
    assert messages[1].text == "notified"
    assert engine.predict_count == 1


# ==== max_function_rounds ====
async def test_full_round_max_function_rounds_forces_final_answer():
    engine = ScriptedEngine([_assistant_call("get_time"), ChatMessage.assistant("Final answer.")])
    ai = OrchestrationKani(engine)
    messages = await _collect_with_max(ai, "go", max_function_rounds=1)

    assert messages[-1].text == "Final answer."
    # first call exposes functions; after hitting the round limit the second call has them disabled
    assert engine.functions_seen == [True, False]


async def test_full_round_warns_when_model_requests_tools_after_max_rounds():
    # the model keeps trying to call tools even after functions are disabled
    engine = ScriptedEngine([_assistant_call("get_time"), _assistant_call("get_time")])
    ai = OrchestrationKani(engine)

    with pytest.warns(UserWarning, match="maximum number of function rounds"):
        messages = await _collect_with_max(ai, "go", max_function_rounds=1)

    # only the first tool call executed; the second (post-limit) request is returned but not run
    assert ai.calls == ["get_time"]
    assert messages[-1].tool_calls  # the trailing message is the unexecuted tool request


async def _collect_with_max(ai: Kani, query, *, max_function_rounds) -> list[ChatMessage]:
    return [msg async for msg in ai.full_round(query, max_function_rounds=max_function_rounds)]


# ==== retry / model feedback on error ====
async def test_full_round_retries_on_retryable_function_error():
    engine = ScriptedEngine([_assistant_call("boom"), ChatMessage.assistant("Recovered.")])
    ai = OrchestrationKani(engine, retry_attempts=1)
    messages = await _collect(ai, "cause an error")

    error_msgs = [m for m in messages if m.role == ChatRole.FUNCTION and m.is_tool_call_error]
    assert len(error_msgs) == 1
    assert "boom" in error_msgs[0].text
    assert messages[-1].text == "Recovered."
    # a retryable error keeps functions available on the retry
    assert engine.functions_seen == [True, True]


async def test_full_round_disables_functions_when_error_is_not_retryable():
    engine = ScriptedEngine([_assistant_call("boom_noretry"), ChatMessage.assistant("Giving up on tools.")])
    ai = OrchestrationKani(engine, retry_attempts=1)
    messages = await _collect(ai, "cause an error")

    assert messages[-1].text == "Giving up on tools."
    # a non-retryable error disables functions on the next turn
    assert engine.functions_seen == [True, False]


async def test_full_round_stops_retrying_after_attempts_exhausted():
    # two errors in a row with only one retry allowed -> the second error disables functions
    engine = ScriptedEngine([_assistant_call("boom"), _assistant_call("boom"), ChatMessage.assistant("done")])
    ai = OrchestrationKani(engine, retry_attempts=1)
    messages = await _collect(ai, "cause errors")

    assert messages[-1].text == "done"
    # call 1: functions on; retry (call 2): functions still on; attempts now exhausted so call 3 disables them
    assert engine.functions_seen == [True, True, False]


# ==== streaming variants ====
async def test_full_round_stream_yields_streams_with_roles():
    engine = ScriptedEngine([_assistant_call("get_time"), ChatMessage.assistant("It is noon.")])
    ai = OrchestrationKani(engine)

    collected = []
    async for stream in ai.full_round_stream("What time is it?"):
        role = stream.role
        async for _ in stream:  # consume tokens
            pass
        collected.append((role, await stream.message()))

    assert [role for role, _ in collected] == [ChatRole.ASSISTANT, ChatRole.FUNCTION, ChatRole.ASSISTANT]
    assert collected[1][1].text == "12:00 noon"
    assert collected[2][1].text == "It is noon."


async def test_full_round_str_yields_assistant_text():
    engine = ScriptedEngine([_assistant_call("get_time"), ChatMessage.assistant("It is noon.")])
    ai = OrchestrationKani(engine)

    outputs = [text async for text in ai.full_round_str("What time is it?")]
    # by default full_round_str yields assistant content; the tool request has no text, the final answer does
    assert "It is noon." in outputs


# ==== chat_round guardrails ====
async def test_chat_round_warns_when_functions_are_defined():
    engine = ScriptedEngine([ChatMessage.assistant("hi")])
    ai = OrchestrationKani(engine)
    with pytest.warns(UserWarning, match="chat_round\\(\\) will not call"):
        await ai.chat_round("hello")


async def test_chat_round_silenced_with_include_functions_false():
    import warnings

    engine = ScriptedEngine([ChatMessage.assistant("hi")])
    ai = OrchestrationKani(engine)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        resp = await ai.chat_round("hello", include_functions=False)
    assert resp.text == "hi"


async def test_full_round_accepts_none_query():
    engine = ScriptedEngine([ChatMessage.assistant("unprompted")])
    ai = OrchestrationKani(engine)
    messages = await _collect(ai, None)
    assert messages[-1].text == "unprompted"
    # a None query adds no user message
    assert all(m.role != ChatRole.USER for m in ai.chat_history)


# ==== handle_function_call_exception (unit) ====
async def test_handle_exception_no_such_function():
    ai = OrchestrationKani(ScriptedEngine([]), retry_attempts=1)
    result = await ai.handle_function_call_exception(
        FunctionCall.with_args("nope"), NoSuchFunction("nope"), attempt=0, tool_call_id="tc1"
    )
    assert isinstance(result, ExceptionHandleResult)
    assert result.should_retry is True  # NoSuchFunction is retryable and attempts remain
    assert result.message.name is None
    assert result.message.tool_call_id == "tc1"
    assert "not defined" in result.message.text


async def test_handle_exception_wrapped_call_exception():
    ai = OrchestrationKani(ScriptedEngine([]), retry_attempts=1)
    err = WrappedCallException(retry=True, original=ValueError("kaboom"))
    result = await ai.handle_function_call_exception(FunctionCall.with_args("f"), err, attempt=0)
    assert result.should_retry is True
    assert result.message.name == "f"
    assert result.message.text == "kaboom"


async def test_handle_exception_no_retry_when_attempts_exhausted():
    ai = OrchestrationKani(ScriptedEngine([]), retry_attempts=1)
    err = WrappedCallException(retry=True, original=ValueError("kaboom"))
    # attempt == retry_attempts -> no more retries
    result = await ai.handle_function_call_exception(FunctionCall.with_args("f"), err, attempt=1)
    assert result.should_retry is False


async def test_handle_exception_no_retry_when_error_not_retryable():
    ai = OrchestrationKani(ScriptedEngine([]), retry_attempts=5)
    err = WrappedCallException(retry=False, original=ValueError("kaboom"))
    result = await ai.handle_function_call_exception(FunctionCall.with_args("f"), err, attempt=0)
    assert result.should_retry is False
