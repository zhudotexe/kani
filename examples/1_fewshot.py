"""
This example shows how you can set the chat_history to accomplish few-shot prompting with kani.

If you haven't already, go take a look at `1_quickstart.py` for an explanation of how to set up kani.
"""

import os

from kani import ChatMessage, Kani, chat_in_terminal
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAIEngine(api_key, model="gpt-4o-mini")

# create a few-shot prompt by providing query-answer pairs
fewshot = [
    ChatMessage.user("thank you"),
    ChatMessage.assistant("arigato"),
    ChatMessage.user("good morning"),
    ChatMessage.assistant("ohayo"),
]
# then supply it to kani when you initialize
ai = Kani(engine, chat_history=fewshot)

if __name__ == "__main__":
    chat_in_terminal(ai)
    # USER: crab
    # ASSISTANT: kani
