"""Example from Advanced Usage docs.

This example shows how kani can be used to easily implement auto-gpt functionality (https://github.com/Significant-Gravitas/Auto-GPT),
which aims to autonomously run GPT.
"""
import json
import os
from typing import Annotated, List

from kani import Kani, chat_in_terminal, ai_function, AIParam
from kani.engines.httpclient import BaseClient
from kani.engines.openai import OpenAIEngine

import aiohttp
import sys

api_key = os.getenv("OPENAI_API_KEY")

engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")


class AutoGPTKani(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_search_results = 5
        self.cached_webpages = {}

    @ai_function()
    async def search(
        self,
        search_terms: Annotated[
            str,
            AIParam(desc='Using the search terms "Train station" will return a list related to "Train station".'),
        ],
    ):
        """Search a list of terms with duckduckgo."""
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise ImportError("Please install duckduckgo-search to use this function.") from e

        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(search_terms):
                if len(results) >= self.max_search_results:
                    break
                results.append(result)
            return json.dumps(results)

    @ai_function()
    async def download(self, url: Annotated[str, AIParam(desc="URL to download contents to disk.")]):
        """Downloads and syncs a webpage to the cache."""

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                text = await response.text()
                self.cached_webpages[url] = text

    @ai_function()
    async def query(self, query: str):
        """Queries cache for information."""

        # Query can be more complex, but for now we'll just do a simple search for matching text
        for url, text in self.cached_webpages.items():
            if query in text:
                return text

        return f"No results found. Try downloading a webpage containing {query} first."

    @ai_function()
    async def write(self, filename: str, text: str):
        """Writes text to disk."""

        with open(filename, "w+") as f:
            f.write(text)

        return "Success!"

    @ai_function()
    async def leave_message(
        self,
        number: Annotated[str, AIParam(desc="Phone number to contact.")],
        message: Annotated[str, AIParam(desc="Message left for the phone number recipient.")],
    ):
        """Calls a phone number and plays a message."""

        # This is a placeholder for a real function that would call a phone number and play a message.
        return f"Message left for {number} saying {message}."


ai = AutoGPTKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai)
