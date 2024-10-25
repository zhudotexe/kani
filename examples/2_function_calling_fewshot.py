import enum
import os
from typing import Annotated

from kani import AIParam, ChatMessage, Kani, ToolCall, ai_function, chat_in_terminal
from kani.engines.openai import OpenAIEngine

api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAIEngine(api_key, model="gpt-4o-mini")


# from 2_function_calling_weather - skip down to L33!
class Unit(enum.Enum):
    FAHRENHEIT = "fahrenheit"
    CELSIUS = "celsius"


class MyKani(Kani):
    @ai_function()
    def get_weather(
        self,
        location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
        unit: Unit,
    ):
        """Get the current weather in a given location."""
        # call some weather API, or just mock it for this example
        degrees = 72 if unit == Unit.FAHRENHEIT else 22
        return f"Weather in {location}: Sunny, {degrees} degrees {unit.value}."


# end 2_function_calling_weather

# build the chat history with examples
fewshot = [
    ChatMessage.user("What's the weather in Philadelphia?"),
    ChatMessage.assistant(
        content=None,
        # use a walrus operator to save a reference to the tool call here...
        tool_calls=[tc := ToolCall.from_function("get_weather", location="Philadelphia, PA", unit="fahrenheit")],
    ),
    # so this function result knows which call it's responding to
    ChatMessage.function("get_weather", "Weather in Philadelphia, PA: Partly cloudy, 85 degrees fahrenheit.", tc.id),
    ChatMessage.assistant(
        content=None,
        tool_calls=[tc2 := ToolCall.from_function("get_weather", location="Philadelphia, PA", unit="celsius")],
    ),
    ChatMessage.function("get_weather", "Weather in Philadelphia, PA: Partly cloudy, 29 degrees celsius.", tc2.id),
    ChatMessage.assistant("It's currently 85F (29C) and partly cloudy in Philadelphia."),
]
# and give it to the kani when you initialize it
ai = MyKani(engine, chat_history=fewshot)

if __name__ == "__main__":
    chat_in_terminal(ai)
