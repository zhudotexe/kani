"""
Usage: python model_test_trains.py hf/model-id

(This file isn't about training models - I just like Japanese trains.)
"""

import asyncio
import json
import sys
from typing import Annotated

import httpx

from kani import AIParam, ChatRole, Kani, ai_function, print_stream, print_width
from kani.utils.cli import create_engine_from_cli_arg
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking

if len(sys.argv) == 2:
    engine = create_engine_from_cli_arg(sys.argv[1])
else:
    print("Usage: python model_test_trains.py provider:model")
    exit(1)


# noinspection DuplicatedCode
class WikipediaRetrievalKani(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wikipedia_client = httpx.AsyncClient(base_url="https://en.wikipedia.org/w/api.php", follow_redirects=True)

    @ai_function()
    async def get_article(
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
    async def search_wikipedia(self, query: str):
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
        " station, its ID, and all the lines (if any) each connects to."
    )

    print("======== testing stream simple ========")
    await stream_query(
        "What are some of the weirdest (real) trains in Japan? When do they operate and how much do they cost?"
    )

    print("======== testing stream complex ========")
    await stream_query(
        "How do I get from Oku-Tama to Noboribetsu without using the Shinkansen?\nOutput a"
        " precise list of steps needed for each route in JSON format as a list of steps. It does not need to be a"
        " real-time route, just an outline of how you would do it. Each step should be of the following"
        ' form:\n```json\n{\n    "from": "Station Name (Station ID)",\n    "to": "Station Name (Station ID)",\n   '
        ' "line": "JR Line Name",\n    "duration": 120, // duration in minutes\n    "cost": 5000 // cost in yen\n}\n```'
    )


# basic system prompt since many models don't include their FC prompt in the chat template...
# system_prompt = """\
# You can use the following functions:
#
# search(query: str) -- Searches for titles of Wikipedia articles.
# wikipedia(title: Annotated[str, AIParam(desc='The article title on Wikipedia, e.g. "Train_station".')]) -- Gets the \
# article text of a Wikipedia article given its title.
# """
system_prompt = "Answer all of the user's questions to the best of your ability with no follow-up questions."

ai = WikipediaRetrievalKani(engine, system_prompt=system_prompt)
if __name__ == "__main__":
    asyncio.run(main())
