"""
Basic chats with no function calling.
"""

import pytest
from pytest_lazy_fixtures import lf

from kani import ChatMessage, Kani, print_stream

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    "engine",
    [
        lf("e2e_anthropic_engine"),
        lf("e2e_google_engine"),
        lf("e2e_openai_engine"),
        lf("e2e_huggingface_engine"),
        lf("e2e_llamacpp_engine"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
class TestE2EChat:
    async def _do_inference(self, ai, query, stream):
        if stream:
            stream = ai.chat_round_stream(query)
            await print_stream(stream)
            msg = await stream.message()
        else:
            msg = await ai.chat_round(query)
            print(msg.text)
        return msg

    async def test_hello(self, engine, stream):
        """Single-round."""
        ai = Kani(engine)
        msg = await self._do_inference(ai, "Hello!", stream)
        assert msg
        assert msg.text

    async def test_system_prompt(self, engine, stream):
        """Single-round, system prompt."""
        ai = Kani(engine, system_prompt="Before replying to the user, output the word 'ABRACADABRA' on its own line.")
        msg = await self._do_inference(ai, "Write me a Python function to calculate a factorial.", stream)
        assert "abracadabra" in msg.text.lower()

    async def test_fewshot(self, engine, stream):
        """Multi-round (few-shot) + system prompt."""
        fewshot = [
            ChatMessage.user("thank you"),
            ChatMessage.assistant("arigato"),
            ChatMessage.user("good morning"),
            ChatMessage.assistant("ohayo"),
        ]
        ai = Kani(
            engine,
            chat_history=fewshot,
            system_prompt=(
                "You are acting as a Japanese translator. Please return the Japanese translation of the user's input in"
                " Romaji only."
            ),
        )
        msg = await self._do_inference(ai, "crab", stream)
        assert "kani" in msg.text.lower()
