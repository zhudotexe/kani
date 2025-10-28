import asyncio
from pathlib import Path

from kani.ext.multimodal_core import ImagePart

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


if __name__ == "__main__":
    asyncio.run(main())
