"""Example from the Customization docs.

This example shows how to override kani's chat prompt building functionality.
"""

import os

from kani import Kani, chat_in_terminal
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAIEngine(api_key, model="gpt-4o-mini")


class LastFourKani(Kani):
    async def get_prompt(self, include_functions=True, **kwargs):
        """
        Only include the most recent 4 messages (omitting earlier ones to fit in the token length if necessary)
        and any always included messages.
        """
        # calculate how many tokens we have for the prompt, accounting for the response
        max_len = self.max_context_size - self.desired_response_tokens
        # try to keep up to the last 4 messages...
        for to_keep in range(4, 0, -1):
            # if the messages fit in the space we have remaining...
            token_len = await self.prompt_token_len(
                messages=self.always_included_messages + self.chat_history[-to_keep:],
                functions=list(self.functions.values()) if include_functions else None,
                **kwargs,
            )
            if token_len <= max_len:
                return self.always_included_messages + self.chat_history[-to_keep:]
        raise ValueError("Could not find a valid prompt including at least 1 message")


ai = LastFourKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai)
