from pathlib import Path

import pytest
from pytest_lazy_fixtures import lf

import kani._optional
from kani import Kani, print_stream, print_width
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.freeze_uuids(side_effect="random"),
]

TEST_DATA_DIR = Path(__file__).parent / "data"


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
@pytest.mark.skipif(not kani._optional.has_multimodal_core, reason="kani-multimodal-core not installed")
class TestE2EMultimodal:
    """Test multimodal inputs."""

    async def _do_inference(self, ai, query, stream):
        from kani.ext.multimodal_core.cli import display_media

        display_media(query, show_text=True)
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

    @pytest.mark.request_model_capabilities(["mm_image"])
    async def test_mm_image(self, engine, stream):
        from kani.ext.multimodal_core import ImagePart

        ai = Kani(engine)
        msg = await self._do_inference(
            ai,
            [
                ImagePart.from_file(TEST_DATA_DIR / "tokyo_station.png"),
                "What building is this an image of, and what city was it taken in?",
            ],
            stream,
        )
        assert msg
        assert msg.text
        assert "tokyo" in msg.text.lower()

    @pytest.mark.request_model_capabilities(["mm_audio"])
    async def test_mm_audio(self, engine, stream):
        from kani.ext.multimodal_core import AudioPart

        ai = Kani(engine)
        msg = await self._do_inference(
            ai,
            [
                AudioPart.from_file(TEST_DATA_DIR / "apollo13.mp3"),
                "What is this a recording of, and why is it notable?",
            ],
            stream,
        )
        assert msg
        assert msg.text
        assert "apollo" in msg.text.lower()

    @pytest.mark.request_model_capabilities(["mm_video"])
    async def test_mm_video(self, engine, stream):
        from kani.ext.multimodal_core import VideoPart

        ai = Kani(engine)
        msg = await self._do_inference(
            ai,
            [
                VideoPart.from_file(TEST_DATA_DIR / "bubble.webm"),
                "What is happening in this video?",
            ],
            stream,
        )
        assert msg
        assert msg.text
        assert "bubble" in msg.text.lower()
