"""Run some tests using local small LMs."""

import logging
import os

import pytest

from kani import ChatMessage, Kani
from kani.engines.huggingface import HuggingEngine

CI_TORCH_DEVICE = os.getenv("CI_TORCH_DEVICE", "cpu")
pytestmark = pytest.mark.local


@pytest.fixture(scope="module")
def model():
    return HuggingEngine("google/gemma-3-1b-it", device=CI_TORCH_DEVICE)


@pytest.fixture()
def create_kani(model):
    def _inner(system_prompt=None, **kwargs):
        return Kani(model, system_prompt=system_prompt, **kwargs)

    return _inner


@pytest.fixture(autouse=True, scope="module")
def debug_logging():
    log = logging.getLogger()
    old_level = log.level
    log.setLevel(logging.DEBUG)
    yield
    log.setLevel(old_level)


async def test_local_model(create_kani, gh_log):
    """Do one round of conversation with the SLM."""
    ai = create_kani()
    print("Q: What are some cool things to do in Tokyo?\n")
    stream = ai.chat_round_stream("What are some cool things to do in Tokyo?")
    async for token in stream:
        print(token, end="", flush=True)
    resp = await stream.message()

    gh_log.write(
        "# One Round\n"
        "*This is a real output from kani running a language model on GitHub Actions.*\n\n"
        "---\n\n"
        "> What are some cool things to do in Tokyo?\n\n"
    )
    gh_log.write(resp.text)
    gh_log.write("\n\n---\n\n")


async def test_chatting_local_models(create_kani, gh_log):
    """Two kanis chatting with each other for 3 rounds."""
    tourist = create_kani(
        system_prompt="You are a tourist with plans to visit Tokyo.",
        chat_history=[
            ChatMessage.user("Hey! I can help you with any questions."),
            ChatMessage.assistant("What are some cool things to do in Tokyo?"),
        ],
    )
    guide = create_kani()

    tourist_response = tourist.chat_history[-1].text
    gh_log.write(
        "# Multi-Round\n"
        "*These are real outputs from two kani running language models on GitHub Actions.*\n\n"
        "---\n\n"
        f"### Tourist\n{tourist_response}\n"
    )
    for _ in range(3):
        print("\n========== GUIDE ==========\n")
        guide_stream = guide.chat_round_stream(tourist_response)
        async for token in guide_stream:
            print(token, end="", flush=True)
        guide_msg = await guide_stream.message()
        guide_response = guide_msg.text

        print("\n========== TOURIST ==========\n")
        tourist_stream = tourist.chat_round_stream(guide_response)
        async for token in tourist_stream:
            print(token, end="", flush=True)
        tourist_msg = await tourist_stream.message()
        tourist_response = tourist_msg.text

        gh_log.write(f"### Guide\n{guide_response}\n\n### Tourist\n{tourist_response}\n\n")
