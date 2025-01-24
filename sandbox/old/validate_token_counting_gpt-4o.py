"""Example from Advanced Usage docs.

This example shows how kani's function calling functionality can be used to retrieve information from an external
database, e.g. Wikipedia.
"""

import json
import logging
import os
from typing import Annotated

import httpx

from kani import AIParam, ChatMessage, Kani, ai_function, chat_in_terminal
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")

engine = OpenAIEngine(api_key, model="gpt-4o")


class WikipediaRetrievalKani(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wikipedia_client = httpx.AsyncClient(base_url="https://en.wikipedia.org/w/api.php", follow_redirects=True)

    async def get_prompt(self) -> list[ChatMessage]:
        msgs = await super().get_prompt()
        print(
            "expected prompt len:",
            sum(self.message_token_len(m) for m in msgs)
            + self.engine.function_token_reserve(list(self.functions.values())),
        )
        return msgs

    async def add_completion_to_history(self, completion):
        print("actual prompt len:", completion.prompt_tokens)
        return await super().add_completion_to_history(completion)

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


ai = WikipediaRetrievalKani(engine)
logging.basicConfig(level=logging.INFO)
logging.getLogger("kani").setLevel(logging.DEBUG)
if __name__ == "__main__":
    chat_in_terminal(ai)
