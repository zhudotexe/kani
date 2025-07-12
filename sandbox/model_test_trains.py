"""
Usage: python model_test_trains.py hf/model-id [tool_call_parser_name [prompt_pipeline_name]]

(This file isn't about training models - I just like Japanese trains.)
"""

import asyncio
import json
import sys
from typing import Annotated

import httpx

from kani import AIParam, ChatRole, Kani, ai_function, print_stream, print_width, tool_parsers
from kani.prompts import impl as prompt_pipelines
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking


# ==== engine defs ====
def chat_openai(model_id: str):
    from kani.engines.openai import OpenAIEngine

    return OpenAIEngine(model=model_id)


def chat_anthropic(model_id: str):
    from kani.engines.anthropic import AnthropicEngine

    return AnthropicEngine(model=model_id)


def chat_google(model_id: str):
    from kani.engines.google import GoogleAIEngine

    return GoogleAIEngine(model=model_id)


def chat_huggingface(model_id: str):
    from kani.engines.huggingface import HuggingEngine

    return HuggingEngine(model_id=model_id, model_load_kwargs={"trust_remote_code": True})


PROVIDER_MAP = {
    # openai
    "openai": chat_openai,
    "oai": chat_openai,
    # anthropic
    "anthropic": chat_anthropic,
    "ant": chat_anthropic,
    "claude": chat_anthropic,
    # google
    "google": chat_google,
    "g": chat_google,
    "gemini": chat_google,
    # huggingface
    "huggingface": chat_huggingface,
    "hf": chat_huggingface,
}


# ==== select engine ====
def get_engine(arg):
    provider, model_id = arg.split("/", 1)
    if provider not in PROVIDER_MAP:
        print(f"Invalid model provider: {provider!r}. Valid options: {list(PROVIDER_MAP)}")
        sys.exit(1)

    return PROVIDER_MAP[provider](model_id)


if len(sys.argv) == 2:
    engine = get_engine(sys.argv[1])
elif len(sys.argv) == 3:
    model = get_engine(sys.argv[1])
    parser_cls = getattr(tool_parsers, sys.argv[2])
    engine = parser_cls(model)
elif len(sys.argv) == 4:
    prompt_pipeline = getattr(prompt_pipelines, sys.argv[3])
    model = get_engine(sys.argv[1])
    parser_cls = getattr(tool_parsers, sys.argv[2])
    engine = parser_cls(model)
else:
    print("Usage: python model_test_trains.py hf/org-id/model-id [tool_call_parser_name [prompt_pipeline_name]]")
    exit(1)


# noinspection DuplicatedCode
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
        " station, its ID, and all the lines (if any) each connects to."
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


# basic system prompt since many models don't include their FC prompt in the chat template...
system_prompt = """\
You can use the following functions:

search(query: str) -- Searches for titles of Wikipedia articles.
wikipedia(title: Annotated[str, AIParam(desc='The article title on Wikipedia, e.g. "Train_station".')]) -- Gets the \
article text of a Wikipedia article given its title.
"""

ai = WikipediaRetrievalKani(engine, system_prompt=system_prompt)
if __name__ == "__main__":
    asyncio.run(main())
