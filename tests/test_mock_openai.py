"""Ensure that any messages sent to OpenAI are valid (mock the API and just echo)."""

import time

from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import RootModel

from kani import ChatMessage, Kani
from kani.engines.openai import OpenAIClient, OpenAIEngine
from kani.engines.openai.models import OpenAIChatMessage


class MockOpenAIClient(OpenAIClient):
    async def request(self, method: str, route: str, headers=None, retry=None, **kwargs):
        if route != "/chat/completions":
            raise ValueError("only chat completions is mocked in tests")

        # validate that all the messages come across correctly
        data = kwargs["json"]
        RootModel[list[OpenAIChatMessage]].model_validate(data["messages"])

    async def post(self, route: str, **kwargs):
        if route != "/chat/completions":
            raise ValueError("only chat completions is mocked in tests")

        await self.request("POST", route, **kwargs)
        data = kwargs["json"]
        message = data["messages"][-1] if data["messages"] else {"role": "assistant", "content": None}
        return dict(
            id="some-id",
            object="chat.completion",
            created=int(time.time()),
            model=data["model"],
            usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            choices=[dict(message=message, index=0)],
        )


class MockOpenAIEngine(OpenAIEngine):
    @staticmethod
    def translate_messages(messages, cls=OpenAIChatMessage):
        # we don't care about the tool call bindings here - just the translation
        return [cls.from_chatmessage(m) for m in messages]


client = MockOpenAIClient("sk-fake-api-key")
engine = MockOpenAIEngine(client=client)


# hypothesis synchronously constructs a coro to call MockOpenAIClient.create_chat_completion
# based on the type annotations of the async function
# we then await the returned coro in the async test body
@settings(suppress_health_check=(HealthCheck.too_slow,), deadline=None)
@given(st.builds(client.create_chat_completion))
async def test_chat_completions_valid(coro):
    await coro


def build_kani_state(msgs: list[ChatMessage]):
    return Kani(engine, chat_history=msgs)


@settings(suppress_health_check=(HealthCheck.too_slow,), deadline=None)
@given(st.builds(build_kani_state))
async def test_kani_chatmessages_valid(ai):
    await ai.get_model_completion()
