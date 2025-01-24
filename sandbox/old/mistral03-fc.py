import enum
from typing import Annotated

from kani import AIParam, Kani, ai_function, chat_in_terminal
from kani.engines.huggingface import HuggingEngine
from kani.prompts.impl.mistral import MISTRAL_V3_PIPELINE, MistralFunctionCallingAdapter

# model = HuggingEngine(model_id="mistralai/Mistral-7B-Instruct-v0.3", prompt_pipeline=MISTRAL_V3_PIPELINE)
model = HuggingEngine(model_id="mistralai/Mistral-Small-Instruct-2409", prompt_pipeline=MISTRAL_V3_PIPELINE)
engine = MistralFunctionCallingAdapter(model)


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


ai = MyKani(engine)
if __name__ == "__main__":
    chat_in_terminal(ai, verbose=True, stream=True)
