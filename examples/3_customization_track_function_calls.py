"""Example from the Customization docs.

This example shows how to override kani's function call handler to add instrumentation.
"""

import collections
import datetime
import os

from kani import Kani, ai_function, chat_in_terminal
from kani.engines.openai import OpenAIEngine
from kani.exceptions import FunctionCallException

api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAIEngine(api_key, model="gpt-4o-mini")


class TrackCallsKani(Kani):
    # You can override __init__ and track kani-specific state:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_calls = collections.Counter()
        self.failed_calls = collections.Counter()

    async def do_function_call(self, call, *args, **kwargs):
        try:
            result = await super().do_function_call(call, *args, **kwargs)
            self.successful_calls[call.name] += 1
            return result
        except FunctionCallException:
            self.failed_calls[call.name] += 1
            raise

    # Let's give the model some functions to work with:
    @ai_function()
    def get_time(self):
        """Get the current time in the user's time zone."""
        # oh no! the clock is broken!
        raise RuntimeError("The time API is currently offline. Please try using `get_date_and_time`.")

    @ai_function()
    def get_date_and_time(self):
        """Get the current day and time in the user's time zone."""
        return str(datetime.datetime.now())


ai = TrackCallsKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai, rounds=1)
    print("Successful:", ai.successful_calls)
    print("Failed:", ai.failed_calls)
