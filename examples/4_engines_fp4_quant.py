"""FP4 quantization compresses models' weights to a 4-bit representation, saving memory on the GPU.

This example shows how to initialize a HuggingFace engine with fp4 quantization enabled on the GPU.

You will need to install ``bitsandbytes`` and ``accelerate`` from pip.
"""

import asyncio
import time

from transformers import BitsAndBytesConfig

from kani import Kani, chat_in_terminal
from kani.engines.huggingface.llama2 import LlamaEngine

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

engine = LlamaEngine(
    use_auth_token=True,
    model_load_kwargs={
        "device_map": "auto",
        "quantization_config": quantization_config,
    },
)
ai = Kani(
    engine,
    system_prompt=(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
        " Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
        " Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not"
        " make any sense, or is not factually coherent, explain why instead of answering something not correct. If you"
        " don't know the answer to a question, please don't share false information."
    ),
)


async def time_completion():
    before = time.monotonic()
    message = await ai.chat_round("What are some interesting things to do in Tokyo?", top_k=1, do_sample=True)
    print(message.content)
    print(f"Tokens: {ai.message_token_len(message)}")
    after = time.monotonic()
    print(f"Time: {after - before}")


if __name__ == "__main__":
    # chat_in_terminal(ai)
    asyncio.run(time_completion())
