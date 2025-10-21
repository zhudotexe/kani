"""
Basic chats with no function calling.
"""

import pytest
from pytest_lazy_fixtures import lf

from kani import ChatMessage, Kani, print_stream, print_width
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking

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
        print_width(query, prefix="USER: ")
        if stream:
            stream = ai.chat_round_stream(query)
            await print_stream(stream, prefix="AI: ")  # todo how to handle reasoning here
            msg = await stream.message()
            text = assistant_message_thinking(msg, show_args=True)
            if text:
                print_width(text, prefix="AI: ")
        else:
            msg = await ai.chat_round(query)
            text = assistant_message_contents_thinking(msg, show_args=True, show_reasoning=True)
            print_width(text, prefix="AI: ")
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

    async def test_last2(self, engine, stream):
        # from examples/3_customization_last_four.py
        # noinspection DuplicatedCode
        class LastTwoKani(Kani):
            async def get_prompt(self, include_functions=True, **kwargs):
                """
                Only include the most recent 4 messages (omitting earlier ones to fit in the token length if necessary)
                and any always included messages.
                """
                # calculate how many tokens we have for the prompt, accounting for the response
                max_len = self.max_context_size - self.desired_response_tokens
                # try to keep up to the last 2 messages...
                for to_keep in range(2, 0, -1):
                    # if the messages fit in the space we have remaining...
                    try:
                        token_len = await self.prompt_token_len(
                            messages=self.always_included_messages + self.chat_history[-to_keep:],
                            functions=list(self.functions.values()) if include_functions else None,
                            **kwargs,
                        )
                        if token_len <= max_len:
                            return self.always_included_messages + self.chat_history[-to_keep:]
                    except Exception as e:
                        print(f"Invalid prompt: {e}")
                        continue
                raise ValueError("Could not find a valid prompt including at least 1 message")

        ai = LastTwoKani(engine)
        await self._do_inference(ai, "Hi! My name is Mizzenmast.", stream)
        await self._do_inference(
            ai, "If you had one, what would be your favorite color? (Please don't mention my name.)", stream
        )
        msg = await self._do_inference(ai, "What is my name? (You can mention it now.)", stream)
        assert "mizzenmast" not in msg.text.lower()
