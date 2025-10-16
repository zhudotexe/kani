import pytest
from pytest_lazy_fixtures import lf

from kani import Kani

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("engine", [lf("e2e_anthropic_engine"), lf("e2e_google_engine"), lf("e2e_openai_engine")])
async def test_hello_api(engine):
    ai = Kani(engine)
    resp = await ai.chat_round_str("Hello!")
    assert resp


async def test_hello_hf(e2e_huggingface_engine):
    ai = Kani(e2e_huggingface_engine)
    resp = await ai.chat_round_str("Hello!")
    assert resp
