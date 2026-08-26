from kani import ChatMessage, ReasoningPart, ToolCall
from kani.utils.message_formatters import (
    all_message_contents,
    assistant_message_contents,
    assistant_message_contents_thinking,
    assistant_message_thinking,
)


def _assistant_with_tools():
    return ChatMessage.assistant(
        content=None,
        tool_calls=[ToolCall.from_function("get_weather", location="SF"), ToolCall.from_function("get_time")],
    )


# ==== all_message_contents ====
def test_all_message_contents_returns_text_for_any_role():
    assert all_message_contents(ChatMessage.user("hi")) == "hi"
    assert all_message_contents(ChatMessage.assistant("yo")) == "yo"
    assert all_message_contents(ChatMessage.function("f", "result")) == "result"


# ==== assistant_message_contents ====
def test_assistant_message_contents_ignores_non_assistant():
    assert assistant_message_contents(ChatMessage.user("hi")) is None


def test_assistant_message_contents_plain_text():
    assert assistant_message_contents(ChatMessage.assistant("the answer")) == "the answer"


def test_assistant_message_contents_reasoning_colored():
    msg = ChatMessage.assistant([ReasoningPart(content="thinking"), "the answer"])
    assert assistant_message_contents(msg, color=True) == "\033[0;37mthinking\033[0m\nthe answer"


def test_assistant_message_contents_reasoning_uncolored():
    msg = ChatMessage.assistant([ReasoningPart(content="thinking"), "the answer"])
    assert assistant_message_contents(msg, color=False) == "thinking\nthe answer"


def test_assistant_message_contents_hides_reasoning_when_disabled():
    msg = ChatMessage.assistant([ReasoningPart(content="thinking"), "the answer"])
    # ReasoningPart stringifies to "" so only the visible text remains
    assert assistant_message_contents(msg, show_reasoning=False) == "the answer"


# ==== assistant_message_thinking ====
def test_assistant_message_thinking_none_without_tool_calls():
    assert assistant_message_thinking(ChatMessage.assistant("hi")) is None
    assert assistant_message_thinking(ChatMessage.user("hi")) is None


def test_assistant_message_thinking_lists_function_names():
    assert assistant_message_thinking(_assistant_with_tools()) == "Thinking... [get_weather; get_time]"


def test_assistant_message_thinking_with_args():
    result = assistant_message_thinking(_assistant_with_tools(), show_args=True)
    assert result == "Thinking... [get_weather(location='SF'); get_time()]"


# ==== assistant_message_contents_thinking ====
def test_contents_thinking_ignores_non_assistant():
    assert assistant_message_contents_thinking(ChatMessage.user("hi")) is None


def test_contents_thinking_text_only():
    assert assistant_message_contents_thinking(ChatMessage.assistant("just text")) == "just text"


def test_contents_thinking_tool_calls_only():
    assert assistant_message_contents_thinking(_assistant_with_tools()) == "Thinking... [get_weather; get_time]"


def test_contents_thinking_text_and_tool_calls():
    msg = ChatMessage.assistant("Let me check.", tool_calls=[ToolCall.from_function("get_time")])
    assert assistant_message_contents_thinking(msg) == "Let me check.\nThinking... [get_time]"
