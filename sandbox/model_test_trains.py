"""
Usage: python model_test_trains.py hf/model-id [tool_call_parser_name]

(This file isn't about training models - I just like Japanese trains.)
"""

import asyncio
import json
import sys
from typing import Annotated

import httpx

from kani import AIParam, ChatRole, Kani, ai_function, print_stream, print_width, tool_parsers
from kani.engines.huggingface import HuggingEngine
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking

if len(sys.argv) == 2:
    engine = HuggingEngine(model_id=sys.argv[1])
elif len(sys.argv) == 3:
    model = HuggingEngine(model_id=sys.argv[1])
    parser_cls = getattr(tool_parsers, sys.argv[2])
    engine = parser_cls(model)
else:
    print("Usage: python model_test_trains.py hf/model-id [tool_call_parser_name]")
    exit(1)


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
    await print_query("How many subway lines does each station on the Yamanote line connect to?")

    print("======== testing stream simple ========")
    await stream_query("What are some of the weirdest trains in Japan?")

    print("======== testing stream complex ========")
    await stream_query(
        "What is the fastest way from Oku-Tama to Noboribetsu? What is the cheapest way? Use JR lines only."
    )


ai = WikipediaRetrievalKani(engine)
if __name__ == "__main__":
    asyncio.run(main())
