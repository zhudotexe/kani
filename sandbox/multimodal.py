import asyncio
import time
from pathlib import Path

from kani.ext.multimodal_core import ImagePart
from transformers import AutoProcessor

from kani import Kani, print_stream
from kani.engines.huggingface import HuggingEngine

TEST_DATA_DIR = Path(__file__).parents[1] / "tests/e2e/data"


async def main():
    engine = HuggingEngine("google/gemma-3-12b-it", max_context_size=128000)
    ai = Kani(engine)
    async for stream in ai.full_round_stream([
        ImagePart.from_file(TEST_DATA_DIR / "tokyo_station.png"),
        "What building is this an image of, and what city was it taken in?",
    ]):
        await print_stream(stream)


async def encode_speed():
    img = ImagePart.from_file(TEST_DATA_DIR / "tokyo_station.png")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ...},
                {"type": "text", "text": "What building is this an image of, and what city was it taken in?"},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    audios = []
    videos = []
    images = [img.image]
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos, add_special_tokens=False, return_tensors="pt"
    )
    return inputs


if __name__ == "__main__":
    start = time.monotonic()
    asyncio.run(main())
    print(time.monotonic() - start)
