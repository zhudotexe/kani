"""
Usage: python model_test_trains.py hf/model-id [tool_call_parser_name [prompt_pipeline_name]]

(This file isn't about training models - I just like Japanese trains.)

NOTE: the cluster can be weird - use this for GPU install (make sure running on a node with a 48GB GPU attached):
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CXX_FLAGS=-fopenmp -DLLAVA_BUILD=OFF" pip install -U llama-cpp-python==0.3.6 --force-reinstall --verbose --no-cache

or the aggressive version:
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CXX_FLAGS=-fopenmp -DLLAVA_BUILD=OFF -DLLAMA_CUBLAS=1 -DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF" pip install -U llama-cpp-python==0.3.6 --force-reinstall --verbose --no-cache
"""

import asyncio
import json
import logging
from typing import Annotated

import httpx

from kani import AIParam, ChatRole, Kani, ai_function, print_stream, print_width
from kani.engines.huggingface import ChatTemplatePromptPipeline
from kani.engines.llamacpp import LlamaCppEngine
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking

logging.basicConfig(level=logging.INFO)

pipeline = ChatTemplatePromptPipeline.from_pretrained("deepseek-ai/DeepSeek-R1")
engine = LlamaCppEngine(
    repo_id="unsloth/DeepSeek-R1-GGUF",
    filename="DeepSeek-R1-Q2_K_XS/DeepSeek-R1-Q2_K_XS-00001-of-00005.gguf",
    prompt_pipeline=pipeline,
    max_context_size=10000,
    model_load_kwargs={"n_gpu_layers": -1, "additional_files": []},
)


class WikipediaRetrievalKani(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wikipedia_client = httpx.AsyncClient(base_url="https://en.wikipedia.org/w/api.php", follow_redirects=True)

    @ai_function()
    async def wikipedia(
        self,
        title: Annotated[str, AIParam(desc='The article title on Wikipedia, e.g. "Train_station".')],
    ):
        """Get additional information about a topic from Wikipedia."""
        # https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles=Train&explaintext=1&formatversion=2
        resp = await self.wikipedia_client.get(
            "/",
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "titles": title,
                "explaintext": 1,
                "formatversion": 2,
            },
        )
        data = resp.json()
        page = data["query"]["pages"][0]
        if extract := page.get("extract"):
            return extract
        return f"The page {title!r} does not exist on Wikipedia."

    @ai_function()
    async def search(self, query: str):
        """Find titles of Wikipedia articles similar to the given query."""
        # https://en.wikipedia.org/w/api.php?action=opensearch&format=json&search=Train
        resp = await self.wikipedia_client.get("/", params={"action": "opensearch", "format": "json", "search": query})
        return json.dumps(resp.json()[1])


async def stream_query(query: str):
    async for stream in ai.full_round_stream(query):
        # assistant
        if stream.role == ChatRole.ASSISTANT:
            await print_stream(stream, prefix="AI: ")
            msg = await stream.message()
            text = assistant_message_thinking(msg, show_args=True)
            if text:
                print_width(text, prefix="AI: ")
        # function
        elif stream.role == ChatRole.FUNCTION:
            msg = await stream.message()
            print_width(msg.text, prefix="FUNC: ")


async def print_query(query: str):
    async for msg in ai.full_round(query):
        # assistant
        if msg.role == ChatRole.ASSISTANT:
            text = assistant_message_contents_thinking(msg, show_args=True)
            print_width(text, prefix="AI: ")
        # function
        elif msg.role == ChatRole.FUNCTION:
            print_width(msg.text, prefix="FUNC: ")


async def main():
    print(engine)
    print("======== testing query simple ========")
    await print_query("Tell me about the Yamanote line.")

    print("======== testing query complex ========")
    await print_query(
        "How many subway lines does each station on the Yamanote line connect to? Give me a precise list of each"
        " station, its ID, and all the lines (if any) each connect to."
    )

    print("======== testing stream simple ========")
    await stream_query(
        "What are some of the weirdest (real) trains in Japan? When do they operate and how much do they cost?"
    )

    print("======== testing stream complex ========")
    await stream_query(
        "What is the fastest way from Oku-Tama to Noboribetsu? What is the cheapest way? Use JR lines only.\nOutput a"
        " precise list of steps needed for each route in JSON format as a list of steps. Each step should be of the"
        " following form:\n"
        "```json\n"
        "{\n"
        '    "from": "Station Name (Station ID)",\n'
        '    "to": "Station Name (Station ID)",\n'
        '    "line": "JR Line Name",\n'
        '    "duration": 120, // duration in minutes\n'
        '    "cost": 5000 // cost in yen\n'
        "}\n```"
    )


ai = WikipediaRetrievalKani(engine)
if __name__ == "__main__":
    asyncio.run(main())
