"""Example from the Customization docs.

This example shows how to override kani's chat prompt building functionality.
"""

import os

from kani import Kani, chat_in_terminal
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAIEngine(api_key, model="gpt-4o-mini")


class LastFourKani(Kani):
    async def get_prompt(self, **kwargs):
        """
        Only include the most recent 4 messages (omitting earlier ones to fit in the token length if necessary)
        and any always included messages.
        """
        # calculate how many tokens we have for the prompt, accounting for the response
        max_len = self.max_context_size - self.desired_response_tokens
        # working backwards through history...
        messages = []
        for message in reversed(self.chat_history[-4:]):
            # if the message fits in the space we have remaining...
            token_len = await self.prompt_token_len(
                messages=self.always_included_messages + [message] + messages,
                functions=list(self.functions.values()),
                **kwargs,
            )
            if token_len <= max_len:
                # add it to the returned prompt!
                messages.insert(0, message)
            else:
                break
        return self.always_included_messages + messages


ai = LastFourKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai)
