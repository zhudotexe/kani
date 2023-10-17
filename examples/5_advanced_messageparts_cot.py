"""Example from Advanced Usage docs.

This example shows how to use the MessagePart API to implement hidden messages, such as chain-of-thought. You can
also use MessageParts in user messages, e.g. to provide images to a multimodal model.

You should be familiar with the concept of Engines before trying to understand this code.
"""

from kani import AIFunction, ChatMessage, Kani, MessagePart, chat_in_terminal
from kani.engines.base import BaseEngine, Completion
from kani.engines.openai import OpenAIEngine


# First, define the message part that will contain the hidden chain of thought data
class ThoughtPart(MessagePart):
    # hold some string data that is the chain-of-thought
    data: str

    # when stringified, this part should be hidden from the user/any engine that does not explicitly support it
    def __str__(self):
        return ""


# Then, define an engine that can use the part we defined - we'll wrap another engine to provide a translation layer
class ChainOfThoughtEngine(BaseEngine):
    def __init__(self, engine: BaseEngine):
        self.engine = engine
        self.max_context_size = engine.max_context_size

    @staticmethod
    def translate_message(message: ChatMessage) -> ChatMessage:
        """Translate a input message into a simple string-only message to pass to the underlying engine."""
        content = ""
        # iterate over parts: if it's a thought, place its data in the content; otherwise prefix it with "Answer: "
        for part in message.parts:
            if isinstance(part, ThoughtPart):
                content += f"{part.data}\n"
            else:
                content += f"Answer: {str(part)}"
        # return the translated message
        return message.copy_with(content=content.strip())

    # === BaseEngine interface ===
    def message_len(self, message: ChatMessage) -> int:
        return self.engine.message_len(self.translate_message(message))

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        # translate the messages
        translated_messages = [self.translate_message(m) for m in messages]

        # generate a completion using the underlying engine given those messages
        result = await self.engine.predict(translated_messages, functions, **hyperparams)

        # parse the string-completion back into parts - with some checks to make sure the model actually output the
        # right format
        text = result.message.text
        if "Answer:" in text:
            thought, answer = text.split("Answer:", 1)
            message_with_parts = result.message.copy_with(parts=[ThoughtPart(data=thought.strip()), answer.strip()])
        else:
            message_with_parts = result.message

        # we'll print the result so you can see the parts, though a real engine wouldn't want to
        print("Model response with parts:", message_with_parts.parts)
        # and return the modified answer
        return Completion(
            message=message_with_parts,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    # additional overrides that pass-through to underlying engine
    def function_token_reserve(self, functions):
        return self.engine.function_token_reserve(functions)

    async def close(self):
        return await self.engine.close()


# Initialize our engine as a wrapper around any underlying engine
cot_engine = ChainOfThoughtEngine(OpenAIEngine())

# Now, use kani as normal - we prompt it to answer with a thought before outputting its final answer with "Answer: "
ai = Kani(
    cot_engine,
    system_prompt=(
        "When answering the user, think step by step. Output your thoughts first, then output the final answer on its"
        ' own line, in the format "Answer: {answer}".'
    ),
)

# In this engine, we print out the parts so you can see what's going on. Normally, you would consume parts in your
# application by iterating over the returned ChatMessage, rather than the engine printing it.
#
# Try this prompt:
# Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
if __name__ == "__main__":
    chat_in_terminal(ai)
