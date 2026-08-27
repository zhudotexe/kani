import importlib
import itertools
from typing import AsyncIterable

import pytest

from kani import PromptPipeline, ReasoningPart
from kani.engines.base import BaseCompletion, Completion
from kani.model_specific import (
    PARSER_REGISTRY,
    PROMPT_PIPELINE_REGISTRY,
    parser_for_hf_model,
    prompt_pipeline_for_hf_model,
)
from kani.model_specific.base import BaseParser
from kani.model_specific.deepseek import DeepSeekR1Parser
from kani.model_specific.gpt_oss import GPTOSSParser
from kani.model_specific.json import NaiveJSONToolCallParser
from kani.model_specific.llama3 import Llama3XToolCallParser
from kani.model_specific.mistral import (
    MISTRAL_V3_PIPELINE,
    MistralToolCallParser,
    ensure_available_tools,
    maybe_json,
)
from kani.model_specific.qwen3 import Qwen3Parser, Qwen3ThinkingParser
from kani.model_specific.qwen3_5 import Qwen3_5Parser
from kani.models import ChatMessage, ChatRole, ToolCall
from kani.prompts.examples import build_conversation, build_functions
from tests.engine import EngineForTests


# ==== registries ====
def test_pipeline_registry():
    for pattern, import_path in PROMPT_PIPELINE_REGISTRY:
        mod_name, attr_name = import_path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        pipe = getattr(mod, attr_name)
        assert pipe


def test_parser_registry():
    for pattern, import_path in PARSER_REGISTRY:
        mod_name, attr_name = import_path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        parser = getattr(mod, attr_name)
        assert parser


# ==== dispatch: parser_for_hf_model / prompt_pipeline_for_hf_model ====
# search_parents=False keeps these offline (parent lookup would hit the HF hub)
def test_parser_for_hf_model_matches_registry_patterns():
    assert parser_for_hf_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") is DeepSeekR1Parser
    assert parser_for_hf_model("meta-llama/Llama-3.1-8B-Instruct") is Llama3XToolCallParser
    assert parser_for_hf_model("openai/gpt-oss-20b") is GPTOSSParser
    assert parser_for_hf_model("mistralai/Mistral-Large-2407") is MistralToolCallParser


def test_parser_for_hf_model_prefers_more_specific_thinking_pattern():
    # the "-Thinking" pattern is registered before the general Qwen3 pattern and must win
    assert parser_for_hf_model("Qwen/Qwen3-8B-Thinking-2507") is Qwen3ThinkingParser
    assert parser_for_hf_model("Qwen/Qwen3-8B") is Qwen3Parser


def test_parser_for_hf_model_dispatches_qwen3_5_family():
    # the Qwen3.5/3.6/3.8 patterns all share the Qwen3_5Parser
    assert parser_for_hf_model("Qwen/Qwen3.5-8B") is Qwen3_5Parser
    assert parser_for_hf_model("Qwen/Qwen3.6-8B") is Qwen3_5Parser
    assert parser_for_hf_model("Qwen/Qwen3.8-8B") is Qwen3_5Parser


def test_parser_for_hf_model_returns_none_when_unknown():
    assert parser_for_hf_model("some-org/totally-unknown-model", search_parents=False) is None


def test_prompt_pipeline_for_hf_model_returns_handwritten_pipeline():
    # patterns whose registry entry is a PromptPipeline instance resolve without a tokenizer
    pipe = prompt_pipeline_for_hf_model("meta-llama/Llama-2-7b-chat-hf")
    assert isinstance(pipe, PromptPipeline)
    mistral_pipe = prompt_pipeline_for_hf_model("mistralai/Mistral-7B-Instruct-v0.3")
    assert isinstance(mistral_pipe, PromptPipeline)


# ==== NaiveJSONToolCallParser (json.py) ====
def test_naive_json_parses_tool_call():
    parser = NaiveJSONToolCallParser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls('{"name": "get_weather", "parameters": {"location": "Tokyo"}}')
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.kwargs == {"location": "Tokyo"}


def test_naive_json_ignores_surrounding_whitespace():
    parser = NaiveJSONToolCallParser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls('  \n {"name": "f", "parameters": {}} \n ')
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "f"


def test_naive_json_valid_json_wrong_shape_is_passthrough():
    # valid JSON that is not a tool call dict should be returned unchanged (the `case other` branch)
    parser = NaiveJSONToolCallParser(EngineForTests())
    for text in ('["not", "a", "call"]', '{"name": "missing_parameters"}', '{"parameters": {}}', "42"):
        content, tool_calls = parser.parse_tool_calls(text)
        assert content == text
        assert tool_calls == []


def test_naive_json_invalid_json_is_passthrough():
    parser = NaiveJSONToolCallParser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls("this is just prose, not json")
    assert content == "this is just prose, not json"
    assert tool_calls == []


# ==== Llama3XToolCallParser (llama3.py) ====
def test_llama3_parses_tool_call_with_python_tag():
    parser = Llama3XToolCallParser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls('<|python_tag|>{"name": "get_weather", "parameters": {"loc": "SF"}}')
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.kwargs == {"loc": "SF"}


def test_llama3_parses_tool_call_without_parameters():
    # the `case {"name": str(name)}` branch: a name with no parameters is a valid no-arg call
    parser = Llama3XToolCallParser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls('{"name": "get_time"}')
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_time"
    assert tool_calls[0].function.kwargs == {}


def test_llama3_valid_json_wrong_shape_is_passthrough():
    parser = Llama3XToolCallParser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls('["not", "a", "call"]')
    assert content == '["not", "a", "call"]'
    assert tool_calls == []


def test_llama3_invalid_json_is_passthrough():
    parser = Llama3XToolCallParser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls("just some text")
    assert content == "just some text"
    assert tool_calls == []


# ==== DeepSeekR1Parser (deepseek.py) ====
# tokens copied from deepseek.py using explicit escapes to avoid transcription errors
_DS_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_DS_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_DS_SEP = "<｜tool▁sep｜>"
_DS_CALL_END = "<｜tool▁call▁end｜>"
_DS_OUTPUTS_END = "<｜tool▁outputs▁end｜>"


def _deepseek_block(name: str, args: str) -> str:
    """Build a DeepSeek tool-call block that the outer + inner regexes will match."""
    return (
        f"{_DS_CALLS_BEGIN}{_DS_CALL_BEGIN}function{_DS_SEP}{name}\n```json\n{args}\n```{_DS_CALL_END}{_DS_OUTPUTS_END}"
    )


def test_deepseek_sanity_tokens_match_source():
    # guards against transcription drift in the escape sequences above
    parser = DeepSeekR1Parser(EngineForTests())
    assert parser.tool_call_start_token == _DS_CALLS_BEGIN
    assert parser.tool_call_end_token == _DS_OUTPUTS_END


def test_deepseek_parses_tool_call_and_trims_content():
    parser = DeepSeekR1Parser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls(
        "Here you go." + _deepseek_block("get_weather", '{"location": "Tokyo"}')
    )
    assert content == "Here you go."
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.kwargs == {"location": "Tokyo"}


def test_deepseek_skips_non_dict_args():
    # decoded args that are valid JSON but not a dict are skipped
    parser = DeepSeekR1Parser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls(_deepseek_block("f", "[1, 2, 3]"))
    assert tool_calls == []


def test_deepseek_skips_undecodable_args():
    parser = DeepSeekR1Parser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls(_deepseek_block("f", "{not valid json}"))
    assert tool_calls == []


def test_deepseek_no_markers_is_passthrough():
    parser = DeepSeekR1Parser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls("no tool calls here")
    assert content == "no tool calls here"
    assert tool_calls == []


# ==== Qwen3Parser (qwen3.py) ====
def test_qwen3_parses_tool_call_and_strips_content():
    parser = Qwen3Parser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls(
        'Let me check.<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>'
    )
    assert content == "Let me check."
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.kwargs == {"location": "Tokyo"}


def test_qwen3_parses_multiple_tool_calls_in_order():
    parser = Qwen3Parser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls(
        '<tool_call>{"name": "first", "arguments": {"a": 1}}</tool_call>'
        '<tool_call>{"name": "second", "arguments": {"b": 2}}</tool_call>'
    )
    assert [tc.function.name for tc in tool_calls] == ["first", "second"]
    assert tool_calls[0].function.kwargs == {"a": 1}
    assert tool_calls[1].function.kwargs == {"b": 2}


def test_qwen3_defaults_missing_arguments_to_empty():
    parser = Qwen3Parser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls('<tool_call>{"name": "get_time"}</tool_call>')
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_time"
    assert tool_calls[0].function.kwargs == {}


def test_qwen3_skips_non_dict_payload():
    parser = Qwen3Parser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls("<tool_call>[1, 2, 3]</tool_call>")
    assert tool_calls == []


def test_qwen3_skips_non_dict_arguments():
    parser = Qwen3Parser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls('<tool_call>{"name": "f", "arguments": "not a dict"}</tool_call>')
    assert tool_calls == []


def test_qwen3_skips_undecodable_tool_call():
    parser = Qwen3Parser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls("<tool_call>{not valid json}</tool_call>")
    assert tool_calls == []


def test_qwen3_thinking_parser_reasons_at_start():
    parser = Qwen3ThinkingParser(EngineForTests())
    assert parser.reasoning_always_at_start is True


# ==== reasoning parsing (BaseParser.parse_reasoning) ====
def test_parse_reasoning_splits_into_part_and_content():
    parser = Qwen3Parser(EngineForTests())  # reasoning tokens <think>/</think>
    result = parser.parse_reasoning("<think>weighing options</think>The answer is 42.")
    assert isinstance(result, list)
    assert isinstance(result[0], ReasoningPart)
    assert result[0].content == "weighing options"
    assert result[1] == "The answer is 42."


def test_parse_reasoning_without_reasoning_is_passthrough():
    parser = Qwen3Parser(EngineForTests())
    assert parser.parse_reasoning("no reasoning here") == "no reasoning here"


def test_qwen3_parse_completion_wires_tool_calls_and_reasoning():
    parser = Qwen3Parser(EngineForTests())
    text = '<think>thinking</think>Let me help.<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'
    completion = parser.parse_completion(Completion(ChatMessage.assistant(text)))
    msg = completion.message
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "f"
    assert isinstance(msg.parts[0], ReasoningPart)
    assert msg.parts[0].content == "thinking"
    assert msg.parts[1] == "Let me help."


# ==== BaseParser.parse_completion exception robustness ====
class _RaisingToolParser(BaseParser):
    def parse_tool_calls(self, content):
        raise RuntimeError("tool parse boom")


class _RaisingReasoningParser(BaseParser):
    def parse_reasoning(self, content):
        raise RuntimeError("reasoning parse boom")


def test_parse_completion_swallows_tool_call_exceptions():
    parser = _RaisingToolParser(EngineForTests())
    completion = parser.parse_completion(Completion(ChatMessage.assistant("hello world")))
    # message is returned unchanged, no tool calls, no exception propagated
    assert completion.message.text == "hello world"
    assert not completion.message.tool_calls


def test_parse_completion_preserves_inner_tool_calls_on_parse_error():
    parser = _RaisingToolParser(EngineForTests())
    msg = ChatMessage.assistant("hello", tool_calls=[ToolCall.from_function("inner")])
    completion = parser.parse_completion(Completion(msg))
    assert [tc.function.name for tc in completion.message.tool_calls] == ["inner"]


def test_parse_completion_swallows_reasoning_exceptions():
    parser = _RaisingReasoningParser(EngineForTests())
    completion = parser.parse_completion(Completion(ChatMessage.assistant("hello world")))
    assert completion.message.text == "hello world"


# ==== MistralToolCallParser (mistral.py) ====
def test_mistral_parses_tool_call_and_trims_content():
    parser = MistralToolCallParser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls(
        'Sure.[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Tokyo"}, "id": "abc123def"}]</s>'
    )
    assert content == "Sure."
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "abc123def"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.kwargs == {"location": "Tokyo"}


def test_mistral_parses_multiple_tool_calls():
    parser = MistralToolCallParser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls(
        '[TOOL_CALLS] [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]</s>'
    )
    assert [tc.function.name for tc in tool_calls] == ["a", "b"]
    assert tool_calls[1].function.kwargs == {"x": 1}


def test_mistral_defaults_missing_arguments_to_empty():
    parser = MistralToolCallParser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls('[TOOL_CALLS] [{"name": "no_args"}]</s>')
    assert len(tool_calls) == 1
    assert tool_calls[0].function.kwargs == {}


def test_mistral_parses_without_end_token():
    # the end token (</s>) is optional in the regex
    parser = MistralToolCallParser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls('[TOOL_CALLS] [{"name": "f", "arguments": {}}]')
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "f"


def test_mistral_no_marker_is_passthrough():
    parser = MistralToolCallParser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls("just a normal response")
    assert content == "just a normal response"
    assert tool_calls == []


# ==== GPTOSSParser (gpt_oss.py) ====
def test_gptoss_parses_reasoning_channel():
    parser = GPTOSSParser(EngineForTests())
    completion = parser.parse_completion(
        Completion(ChatMessage.assistant("<|channel|>analysis<|message|>let me think<|end|>"))
    )
    assert not completion.message.tool_calls
    assert isinstance(completion.message.parts[0], ReasoningPart)
    assert completion.message.parts[0].content == "let me think"


def test_gptoss_parses_final_channel_as_content():
    parser = GPTOSSParser(EngineForTests())
    completion = parser.parse_completion(
        Completion(ChatMessage.assistant("<|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>"))
    )
    assert not completion.message.tool_calls
    assert completion.message.parts == ["The answer is 42."]


def test_gptoss_parses_tool_call_from_functions_recipient():
    parser = GPTOSSParser(EngineForTests())
    completion = parser.parse_completion(
        Completion(
            ChatMessage.assistant(
                '<|channel|>commentary to=functions.get_weather<|message|>{"location": "Tokyo"}<|call|>'
            )
        )
    )
    assert len(completion.message.tool_calls) == 1
    assert completion.message.tool_calls[0].function.name == "get_weather"
    assert completion.message.tool_calls[0].function.kwargs == {"location": "Tokyo"}


def test_gptoss_parses_reasoning_and_tool_call_together():
    parser = GPTOSSParser(EngineForTests())
    text = (
        "<|channel|>analysis<|message|>thinking<|end|>"
        '<|start|>assistant<|channel|>commentary to=functions.f<|message|>{"x": 1}<|call|>'
    )
    completion = parser.parse_completion(Completion(ChatMessage.assistant(text)))
    assert isinstance(completion.message.parts[0], ReasoningPart)
    assert completion.message.parts[0].content == "thinking"
    assert [tc.function.name for tc in completion.message.tool_calls] == ["f"]


# ==== BaseParser streaming (base.py) ====
class _ScriptedStreamEngine(EngineForTests):
    """A mock engine that yields a predetermined list of string chunks when streamed.

    If ``final_completion`` is provided, it is yielded last (as engines that report their own completion do).
    """

    def __init__(self, chunks: list[str], final_completion: BaseCompletion = None, **kwargs):
        super().__init__(**kwargs)
        self.chunks = chunks
        self.final_completion = final_completion

    async def stream(self, messages, functions=None, **hyperparams) -> AsyncIterable[str | BaseCompletion]:
        for chunk in self.chunks:
            yield chunk
        if self.final_completion is not None:
            yield self.final_completion


async def _collect_stream(parser) -> tuple[list[str], BaseCompletion | None]:
    yielded = []
    completion = None
    async for elem in parser.stream([ChatMessage.user("hi")]):
        if isinstance(elem, str):
            yielded.append(elem)
        else:
            completion = elem
    return yielded, completion


async def test_base_stream_hides_reasoning_and_tool_tokens_by_default():
    chunks = [
        "<think>", "secret thoughts", "</think>",
        "Hello ", "world ",
        "<tool_call>", '{"name": "f", "arguments": {}}', "</tool_call>",
    ]  # fmt: skip
    parser = Qwen3Parser(_ScriptedStreamEngine(chunks, max_context_size=1000))
    yielded, completion = await _collect_stream(parser)

    # reasoning and tool-call tokens are buffered out of the stream by default
    assert "".join(yielded) == "Hello world "
    # but the final completion still parses them out of the accumulated content
    assert [tc.function.name for tc in completion.message.tool_calls] == ["f"]
    reasoning = [p for p in completion.message.parts if isinstance(p, ReasoningPart)]
    assert reasoning and reasoning[0].content == "secret thoughts"


async def test_base_stream_shows_reasoning_when_requested():
    chunks = ["<think>", "visible reasoning", "</think>", "answer"]
    parser = Qwen3Parser(_ScriptedStreamEngine(chunks, max_context_size=1000), show_reasoning_in_stream=True)
    yielded, _ = await _collect_stream(parser)
    joined = "".join(yielded)
    assert "visible reasoning" in joined
    assert "answer" in joined


async def test_base_stream_emits_leading_color_when_reasoning_at_start():
    # a reasoning_always_at_start parser opens the stream with the gray ANSI color code
    chunks = ["thinking out loud", "</think>", "the answer"]
    parser = Qwen3ThinkingParser(_ScriptedStreamEngine(chunks, max_context_size=1000), show_reasoning_in_stream=True)
    yielded, _ = await _collect_stream(parser)
    assert yielded[0] == "\033[0;37m"
    assert "thinking out loud" in "".join(yielded)


async def test_base_stream_uses_engine_reported_completion():
    # when the engine reports its own completion, the parser parses that rather than rebuilding from tokens
    inner = Completion(ChatMessage.assistant('done<tool_call>{"name": "f", "arguments": {}}</tool_call>'))
    engine = _ScriptedStreamEngine(
        ["done", "<tool_call>", '{"name": "f", "arguments": {}}', "</tool_call>"],
        final_completion=inner,
        max_context_size=1000,
    )
    _, completion = await _collect_stream(Qwen3Parser(engine))
    assert completion is not None
    assert [tc.function.name for tc in completion.message.tool_calls] == ["f"]
    assert completion.message.text == "done"


def test_parse_completion_concatenates_parser_and_inner_tool_calls_with_warning():
    parser = Qwen3Parser(EngineForTests())
    msg = ChatMessage.assistant(
        '<tool_call>{"name": "parsed", "arguments": {}}</tool_call>',
        tool_calls=[ToolCall.from_function("inner")],
    )
    with pytest.warns(UserWarning, match="Both the tool parser"):
        completion = parser.parse_completion(Completion(msg))
    names = [tc.function.name for tc in completion.message.tool_calls]
    assert names == ["inner", "parsed"]


# ==== Qwen3_5Parser (qwen3_5.py) — <function=name>/<parameter=name> XML-ish format ====
def test_qwen3_5_parses_tool_call():
    parser = Qwen3_5Parser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls(
        "Let me check.<tool_call><function=get_weather><parameter=location>\nTokyo\n</parameter></function></tool_call>"
    )
    assert content == "Let me check."
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    # a value that isn't valid JSON is kept as a raw string
    assert tool_calls[0].function.kwargs == {"location": "Tokyo"}


def test_qwen3_5_parses_multiple_parameters():
    parser = Qwen3_5Parser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls(
        "<tool_call><function=f><parameter=a>1</parameter><parameter=b>two</parameter></function></tool_call>"
    )
    # JSON-decodable values (1) are decoded; non-JSON values (two) stay raw strings
    assert tool_calls[0].function.kwargs == {"a": 1, "b": "two"}


def test_qwen3_5_json_decodes_object_like_parameter_values():
    parser = Qwen3_5Parser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls(
        "<tool_call><function=f>"
        "<parameter=count>3</parameter>"
        "<parameter=ratio>1.5</parameter>"
        "<parameter=enabled>true</parameter>"
        "<parameter=items>[1, 2, 3]</parameter>"
        '<parameter=meta>{"k": "v"}</parameter>'
        "</function></tool_call>"
    )
    assert tool_calls[0].function.kwargs == {
        "count": 3,
        "ratio": 1.5,
        "enabled": True,
        "items": [1, 2, 3],
        "meta": {"k": "v"},
    }


def test_qwen3_5_parses_multiple_tool_calls_in_order():
    parser = Qwen3_5Parser(EngineForTests())
    _, tool_calls = parser.parse_tool_calls(
        "<tool_call><function=first><parameter=x>1</parameter></function></tool_call>"
        "<tool_call><function=second><parameter=y>2</parameter></function></tool_call>"
    )
    assert [tc.function.name for tc in tool_calls] == ["first", "second"]


def test_qwen3_5_skips_tool_call_without_function_tag():
    parser = Qwen3_5Parser(EngineForTests())
    content, tool_calls = parser.parse_tool_calls("<tool_call>no function tag here</tool_call>")
    assert tool_calls == []


def test_qwen3_5_parse_one_tool_call_returns_none_when_invalid():
    parser = Qwen3_5Parser(EngineForTests())
    assert parser.parse_one_tool_call("not a function call") is None


def test_qwen3_5_duplicate_parameter_takes_last_value():
    parser = Qwen3_5Parser(EngineForTests())
    tc = parser.parse_one_tool_call(
        "<function=f><parameter=x>first</parameter><parameter=x>second</parameter></function>"
    )
    assert tc is not None
    assert tc.function.kwargs == {"x": "second"}


# ==== GPTOSSParser streaming (gpt_oss.py state machine) ====
async def test_gptoss_stream_hides_reasoning_by_default():
    chunks = [
        "<|channel|>analysis<|message|>",
        "secret reasoning",
        "<|end|>",
        "<|start|>assistant<|channel|>final<|message|>",
        "the answer",
        "<|end|>",
    ]
    parser = GPTOSSParser(_ScriptedStreamEngine(chunks, max_context_size=1000))
    yielded, completion = await _collect_stream(parser)

    # only the final channel content is visible in the stream
    assert "".join(yielded) == "the answer"
    # the final completion still separates reasoning from content
    reasoning = [p for p in completion.message.parts if isinstance(p, ReasoningPart)]
    assert reasoning and reasoning[0].content == "secret reasoning"


async def test_gptoss_stream_shows_reasoning_when_requested():
    chunks = [
        "<|channel|>analysis<|message|>",
        "visible reasoning",
        "<|end|>",
        "<|start|>assistant<|channel|>final<|message|>",
        "the answer",
        "<|end|>",
    ]
    parser = GPTOSSParser(_ScriptedStreamEngine(chunks, max_context_size=1000), show_reasoning_in_stream=True)
    yielded, _ = await _collect_stream(parser)
    joined = "".join(yielded)
    assert "visible reasoning" in joined
    assert "the answer" in joined


async def test_gptoss_stream_hides_tool_call_content():
    chunks = [
        "<|start|>assistant<|channel|>commentary to=functions.f<|message|>",
        '{"x": 1}',
        "<|call|>",
    ]
    parser = GPTOSSParser(_ScriptedStreamEngine(chunks, max_context_size=1000))
    yielded, completion = await _collect_stream(parser)

    # tool-call arguments (to=functions.*) are never yielded to the user
    assert "".join(yielded) == ""
    assert [tc.function.name for tc in completion.message.tool_calls] == ["f"]


# ==== NaiveJSONToolCallParser streaming (json.py) ====
async def test_naive_json_stream_buffers_leading_brace():
    chunks = ["{", '"name": "f", "parameters": {}}']
    parser = NaiveJSONToolCallParser(_ScriptedStreamEngine(chunks, max_context_size=1000))
    yielded, completion = await _collect_stream(parser)

    # a message that opens with `{` is buffered as a potential tool call, not streamed
    assert "".join(yielded) == ""
    assert [tc.function.name for tc in completion.message.tool_calls] == ["f"]


async def test_naive_json_stream_yields_plain_text():
    chunks = ["Hello ", "world"]
    parser = NaiveJSONToolCallParser(_ScriptedStreamEngine(chunks, max_context_size=1000))
    yielded, completion = await _collect_stream(parser)
    assert "".join(yielded) == "Hello world"
    assert not completion.message.tool_calls


async def test_naive_json_stream_does_not_buffer_brace_after_text():
    # once a non-tool-call token has been seen, a later `{` should still stream through
    chunks = ["Here: ", "{not a call}"]
    parser = NaiveJSONToolCallParser(_ScriptedStreamEngine(chunks, max_context_size=1000))
    yielded, _ = await _collect_stream(parser)
    assert "".join(yielded) == "Here: {not a call}"


# ==== MistralToolCallParser predict/stream end-token stripping (mistral.py) ====
class _FixedPredictEngine(EngineForTests):
    """A mock engine whose predict() returns a fixed assistant message."""

    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self._text = text

    async def predict(self, messages, functions=None, **hyperparams) -> Completion:
        return Completion(ChatMessage.assistant(self._text))


async def test_mistral_predict_strips_end_token():
    parser = MistralToolCallParser(_FixedPredictEngine("The answer.</s>", max_context_size=1000))
    completion = await parser.predict([ChatMessage.user("hi")])
    assert completion.message.text == "The answer."


async def test_mistral_stream_strips_end_token():
    chunks = ["The answer.", "</s>"]
    parser = MistralToolCallParser(_ScriptedStreamEngine(chunks, max_context_size=1000))
    yielded, completion = await _collect_stream(parser)
    assert "".join(yielded) == "The answer."
    assert completion.message.text == "The answer."


async def test_mistral_stream_parses_tool_call():
    chunks = ["Sure.", "[TOOL_CALLS]", ' [{"name": "f", "arguments": {}}]', "</s>"]
    parser = MistralToolCallParser(_ScriptedStreamEngine(chunks, max_context_size=1000))
    yielded, completion = await _collect_stream(parser)
    assert "".join(yielded) == "Sure."
    assert [tc.function.name for tc in completion.message.tool_calls] == ["f"]


# ==== MISTRAL_V3_PIPELINE prompt building (mistral.py) ====
def test_mistral_pipeline_renders_tool_calling_prompt():
    messages = list(itertools.chain.from_iterable(build_conversation(function_call=True)))
    functions = build_functions(function_call=True)
    prompt = MISTRAL_V3_PIPELINE(messages, functions)

    # the available tools are advertised, and the tool call + result use the Mistral delimiters
    assert "[AVAILABLE_TOOLS]" in prompt and "[/AVAILABLE_TOOLS]" in prompt
    assert "get_weather" in prompt
    assert "[TOOL_CALLS]" in prompt
    assert "[TOOL_RESULTS]" in prompt and "[/TOOL_RESULTS]" in prompt


def test_maybe_json_decodes_json_and_passes_through_prose():
    assert maybe_json('{"temp": 21}') == {"temp": 21}
    assert maybe_json("just prose") == "just prose"


def test_ensure_available_tools_prepends_when_no_messages():
    functions = build_functions(function_call=True)
    out = ensure_available_tools([], functions)
    assert len(out) == 1
    assert out[0].role == ChatRole.USER
    assert "[AVAILABLE_TOOLS]" in out[0].text


def test_ensure_available_tools_noop_without_functions():
    assert ensure_available_tools([], []) == []
