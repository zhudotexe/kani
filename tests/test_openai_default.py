import json

import pytest

from kani.ai_function import AIFunction
from kani.models import ChatMessage

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")
pytest.importorskip("tiktoken")

from kani.engines.openai import OpenAIEngine, engine as engine_module
from kani.engines.openai.mm_tokens import tokens_from_image_size


@pytest.mark.parametrize(
    ("base_url", "override", "expected_model", "context_size"),
    [
        ("https://api.openai.com/v1", {}, "gpt-5.6-luna", 1_050_000),
        ("https://api.openai.com/v1", {"model": "gpt-4o-mini"}, "gpt-4o-mini", 128_000),
        ("https://gateway.example/v1", {}, "gpt-4.1-nano", 1_047_576),
    ],
)
async def test_openai_chat_default_on_the_wire(base_url, override, expected_model, context_size):
    def echo(text: str) -> str:
        """Echo a message."""
        return text

    def respond(request):
        assert str(request.url) == f"{base_url}/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == expected_model
        assert body["messages"] == [{"role": "user", "content": "Echo hello"}]
        assert body["tools"][0]["function"]["name"] == "echo"
        assert body["max_completion_tokens"] == 64
        if expected_model == "gpt-5.6-luna":
            assert body["reasoning_effort"] == "none"
        else:
            assert "reasoning_effort" not in body
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": expected_model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_echo",
                                    "type": "function",
                                    "function": {"name": "echo", "arguments": '{"text":"hello"}'},
                                },
                            ],
                        },
                    },
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        )

    async with openai.AsyncOpenAI(
        api_key="synthetic", base_url=base_url, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        engine = OpenAIEngine(client=client, api_type="chat_completions", **override)
        result = await engine.predict([ChatMessage.user("Echo hello")], [AIFunction(echo)], max_completion_tokens=64)

    assert engine.max_context_size == context_size
    assert result.message.tool_calls[0].function.name == "echo"
    assert result.message.tool_calls[0].function.kwargs == {"text": "hello"}


async def test_default_endpoint_and_custom_base_without_injected_client(monkeypatch):
    def build_client(**kwargs):
        return openai.AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(trust_env=False))

    monkeypatch.setattr(engine_module, "OpenAIClient", build_client)
    for kwargs, model in [
        ({}, "gpt-5.6-luna"),
        ({"api_base": "https://gateway.example/v1"}, "gpt-4.1-nano"),
    ]:
        engine = OpenAIEngine(api_key="synthetic", api_type="chat_completions", **kwargs)
        try:
            assert engine.model == model
        finally:
            await engine.close()


@pytest.mark.parametrize(
    ("size", "low_detail", "tokens"), [((65, 32), False, 4), ((2048, 1024), False, 2458), ((1024, 256), True, 77)]
)
def test_luna_image_patch_estimate(size, low_detail, tokens):
    assert tokens_from_image_size(size, "gpt-5.6-luna", low_detail=low_detail) == tokens
