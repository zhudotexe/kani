import asyncio
from typing import Literal, overload

import aiohttp
import pydantic

from chatterbox.models import ChatMessage, FunctionSpec
from .models import SpecificFunctionCall, ChatCompletion, Completion
from ..httpclient import BaseClient, HTTPException, HTTPStatusException, HTTPTimeout


class OpenAIClient(BaseClient):
    SERVICE_BASE = "https://api.openai.com/v1"

    def __init__(self, http: aiohttp.ClientSession, api_key: str):
        super().__init__(http)
        self.api_key = api_key

    async def request(self, method: str, route: str, headers=None, retry=5, **kwargs):
        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self.api_key}"
        for i in range(retry):
            try:
                return await super().request(method, route, headers=headers, **kwargs)
            except (HTTPStatusException, HTTPTimeout) as e:
                if (i + 1) == retry:
                    raise
                retry_sec = 2**i
                self.logger.warning(f"OpenAI returned {e}, retrying in {retry_sec} sec...")
                await asyncio.sleep(retry_sec)
        raise RuntimeError("ran out of retries but no error encountered, halp")

    # ==== completions ====
    @overload
    async def create_completion(
        self,
        model: str,
        prompt: str = "<|endoftext|>",
        suffix: str = None,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        logprobs: int = None,
        echo: bool = False,
        stop: str | list[str] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        best_of: int = 1,
        logit_bias: dict = None,
        user: str = None,
    ) -> Completion:
        ...

    async def create_completion(self, model: str, **kwargs) -> Completion:
        data = await self.post("/completions", json={"model": model, **kwargs})
        try:
            return Completion.model_validate(data)
        except pydantic.ValidationError as e:
            self.logger.exception(f"Failed to deserialize OpenAI response: {data}")
            raise HTTPException(f"Could not deserialize response: {data}") from e

    # ==== chat ====
    @overload
    async def create_chat_completion(
        self,
        model: str,
        messages: list[ChatMessage],
        functions: list[FunctionSpec] | None = None,
        function_call: SpecificFunctionCall | Literal["auto"] | Literal["none"] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stop: str | list[str] | None = None,
        max_tokens: int | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: dict | None = None,
        user: str | None = None,
    ) -> ChatCompletion:
        ...

    async def create_chat_completion(self, model: str, messages: list[ChatMessage], **kwargs) -> ChatCompletion:
        # transform pydantic models
        if "functions" in kwargs:
            kwargs["functions"] = [f.model_dump() for f in kwargs["functions"]]
        if "function_call" in kwargs and isinstance(kwargs["function_call"], SpecificFunctionCall):
            kwargs["function_call"] = kwargs["function_call"].model_dump()
        # call API
        data = await self.post(
            "/chat/completions", json={"model": model, "messages": [cm.model_dump() for cm in messages], **kwargs}
        )
        try:
            return ChatCompletion.model_validate(data)
        except pydantic.ValidationError:
            self.logger.exception(f"Failed to deserialize OpenAI response: {data}")
            raise HTTPException(f"Could not deserialize response: {data}")
