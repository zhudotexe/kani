import json
from typing import Annotated

import pydantic

from kani import AIParam, FunctionCall, Kani, ToolCall, ai_function, chat_in_terminal
from kani.engines import WrapperEngine
from kani.engines.huggingface import HuggingEngine


class NaiveLlamaJSONFunctionCallingEngine(WrapperEngine):
    async def predict(self, *args, **kwargs):
        completion = await self.engine.predict(*args, **kwargs)
        # if the completion is only JSON, try parsing it as a function call
        try:
            data = json.loads(completion.message.text)
            function_call = LlamaFunctionCall.model_validate(data)
        except (json.JSONDecodeError, pydantic.ValidationError):
            return completion
        else:
            tc = ToolCall.from_function_call(FunctionCall.with_args(function_call.name, **function_call.parameters))
            completion.message.content = None
            completion.message.tool_calls = [tc]
        return completion


class LlamaFunctionCall(pydantic.BaseModel):
    name: str
    parameters: dict


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
    engine = NaiveLlamaJSONFunctionCallingEngine(HuggingEngine(model_id="meta-llama/Llama-3.1-70B-Instruct"))
    ai = MyKani(engine)
    chat_in_terminal(ai, verbose=True, stream=False)
