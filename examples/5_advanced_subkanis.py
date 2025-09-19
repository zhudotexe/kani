"""Example from Advanced Usage docs.

This example shows how kani can be used in conjunction with function calling to spawn sub-kani. For example, you might
give a sub-kani a different set of functions, use a more powerful engine, or perform an isolated task.

In this example, we show how a sub-kani can use a different engine to perform summarization, then give the results
to the parent kani.
"""

import os

from kani import Kani, ai_function, chat_in_terminal
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")

fast_model = "gpt-4o-mini"
long_context_model = "gpt-3.5-turbo-16k"


class KaniWithAISummarization(Kani):
    @ai_function()
    async def summarize_conversation(self):
        """Get the summary of the conversation so far."""
        # in this AI Function, we can spawn a sub-kani with a model that can handle
        # longer contexts, since the conversation may be longer than the fast model's
        # context window
        long_context_engine = OpenAIEngine(api_key, model=long_context_model)
        # first, copy the parent's chat history to the child, except the last user message
        # and the function call ([:-2])
        sub_kani = Kani(long_context_engine, chat_history=self.chat_history[:-2])
        # then we ask it to summarize the whole thing, and return the result to the parent kani
        return await sub_kani.chat_round_str("Please summarize the conversation so far.")


fast_engine = OpenAIEngine(api_key, fast_model)
ai = KaniWithAISummarization(fast_engine)
if __name__ == "__main__":
    chat_in_terminal(ai)
