"""Example from Advanced Usage docs.

This example shows how kani's function calling functionality can be used to retrieve information from an external
database, e.g. Wikipedia.
"""

import json
import os
from typing import Annotated

from kani import AIParam, Kani, ai_function, chat_in_terminal
from kani.engines.httpclient import BaseClient
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")

engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")


# let's define a client using kani's BaseClient:
class WikipediaClient(BaseClient):
    SERVICE_BASE = "https://en.wikipedia.org/w/api.php"


class WikipediaRetrievalKani(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wikipedia_client = WikipediaClient()

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
        page = resp["query"]["pages"][0]
        if extract := page.get("extract"):
            return extract
        return f"The page {title!r} does not exist on Wikipedia."

    @ai_function()
    async def search(self, query: str):
        """Find titles of Wikipedia articles similar to the given query."""
        # https://en.wikipedia.org/w/api.php?action=opensearch&format=json&search=Train
        resp = await self.wikipedia_client.get("/", params={"action": "opensearch", "format": "json", "search": query})
        return json.dumps(resp[1])


ai = WikipediaRetrievalKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai)
