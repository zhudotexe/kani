import json
from typing import Annotated

from kani import AIParam, Kani, ai_function, chat_in_terminal
from kani.engines.anthropic import AnthropicEngine


class MyKani(Kani):
    @ai_function()
    def get_weather(
        self,
        location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
        unit: Annotated[str, AIParam(desc="'f' or 'c'")],
    ):
        """Get the current weather in a given location."""
        # call some weather API, or just mock it for this example
        degrees = 72 if unit == "f" else 22
        return json.dumps({"output": f"Weather in {location}: Sunny, {degrees} degrees {unit}."})


if __name__ == "__main__":
    engine = AnthropicEngine(model="claude-3-5-haiku-20241022")
    ai = MyKani(engine)
    chat_in_terminal(ai, verbose=True, stream=True)
