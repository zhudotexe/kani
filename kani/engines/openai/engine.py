import tiktoken

from kani.models import ChatMessage, FunctionSpec
from .client import OpenAIClient
from .models import ChatCompletion
from ..base import BaseEngine

# https://platform.openai.com/docs/models
CONTEXT_SIZES_BY_PREFIX = [
    ("gpt-3.5-turbo-16k", 16384),
    ("gpt-3.5-turbo", 4096),
    ("gpt-4-32k", 32768),
    ("gpt-4", 8192),
    ("text-davinci-", 4096),
    ("code-", 8000),
    ("", 2048),  # e.g. aba/babbage/curie/davinci
]


class OpenAIEngine(BaseEngine):
    def __init__(self, api_key: str, model="gpt-3.5-turbo", max_context_size: int = None, **hyperparams):
        if max_context_size is None:
            max_context_size = next(size for prefix, size in CONTEXT_SIZES_BY_PREFIX if model.startswith(prefix))
        self.client = OpenAIClient(api_key)
        self.model = model
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def message_len(self, message: ChatMessage) -> int:
        mlen = len(self.tokenizer.encode(message.content)) + 5  # ChatML = 4, role = 1
        if message.name:
            mlen += len(self.tokenizer.encode(message.name))
        if message.function_call:
            mlen += len(self.tokenizer.encode(message.function_call.name))
            mlen += len(self.tokenizer.encode(message.function_call.arguments))
        return mlen

    async def predict(
        self, messages: list[ChatMessage], functions: list[FunctionSpec] | None = None, **hyperparams
    ) -> ChatCompletion:
        return await self.client.create_chat_completion(
            model=self.model, messages=messages, functions=functions, **self.hyperparams, **hyperparams
        )

    async def close(self):
        await self.client.close()
