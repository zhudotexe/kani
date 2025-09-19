import os

from kani import Kani, chat_in_terminal
from kani.engines.openai import OpenAIEngine

# Replace this with your OpenAI API key (https://platform.openai.com/account/api-keys)
# or run `export OPENAI_API_KEY="sk-..."` in your terminal.
api_key = os.getenv("OPENAI_API_KEY")

# kani uses an Engine to interact with the language model. You can specify other model parameters here,
# like temperature=0.7.
engine = OpenAIEngine(api_key, model="gpt-4o-mini")

# The kani manages the chat state, prompting, and function calling. Here, we only give it the engine to call
# ChatGPT, but you can specify other parameters like system_prompt="You are..." here.
ai = Kani(engine)

# kani comes with a utility to interact with a kani through your terminal! Check out the docs for how to use kani
# programmatically.
if __name__ == "__main__":
    chat_in_terminal(ai)
