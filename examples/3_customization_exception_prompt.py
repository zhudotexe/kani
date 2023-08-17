"""Example from the Customization docs.

This example shows how to override kani's function call exception handler to use a custom prompt.
"""
import os

from kani import ChatMessage, Kani, ai_function, chat_in_terminal
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")


class CustomExceptionPromptKani(Kani):
    async def handle_function_call_exception(self, call, err, attempt):
        self.chat_history.append(
            ChatMessage.system(
                f"The call encountered an error. Relay this error message to the user in a sarcastic manner: {err}"
            )
        )
        return attempt < self.retry_attempts and err.retry

    @ai_function()
    def get_time(self):
        """Get the current time in the user's time zone."""
        raise RuntimeError("The time API is currently offline (error 0xDEADBEEF).")


ai = CustomExceptionPromptKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai)
