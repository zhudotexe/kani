from kani.engines.base import BaseEngine
from kani.models import ChatMessage, ChatRole
from kani.ai_function import AIFunction
from kani.exceptions import KaniException
import httpx
from typing import List, Optional, AsyncGenerator
import json
import logging
from urllib.parse import urlparse

class CerebrasCompletion:
    def __init__(self, text: str, function_call: Optional[dict] = None):
        self.text = text
        self.function_call = function_call

    @property
    def message(self) -> ChatMessage:
        if self.function_call:
            return ChatMessage(role=ChatRole.ASSISTANT, content=self.text, function_call=self.function_call)
        return ChatMessage(role=ChatRole.ASSISTANT, content=self.text)

class CerebrasEngine(BaseEngine):
    def __init__(
        self,
        api_key: str,
        model: str = "llama3.1-8b",
        api_base: str = "https://api.cerebras.net/v1",
        max_tokens: int = 1000,
        max_context_size: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.max_context_size = max_context_size
        self.temperature = temperature
        self.client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0
        )
        self.kwargs = kwargs

    async def close(self):
        await self.client.aclose()

    def message_len(self, message: ChatMessage) -> int:
        return len(message.content) // 4

    async def predict(
        self,
        messages: List[ChatMessage],
        functions: Optional[List[AIFunction]] = None,
        **kwargs
    ) -> CerebrasCompletion:
        data = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
        }

        if functions:
            data["functions"] = [f.to_dict() for f in functions]

        try:
            response = await self.client.post("/chat/completions", json=data)
            await response.aread()  # Ensure the entire response is read
            response.raise_for_status()
            result = await response.json()  # Add await here

            choice = result["choices"][0]
            text = choice["message"]["content"]
            function_call = choice["message"].get("function_call")

            return CerebrasCompletion(text, function_call)
        except httpx.HTTPStatusError as e:
            raise KaniException(f"Cerebras API returned status code {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise KaniException(f"Error communicating with Cerebras API: {str(e)}")
        except Exception as e:
            logging.exception(f"Unexpected error in predict: {e}")
            raise KaniException(f"Unexpected error: {str(e)}")

    async def stream(
        self,
        messages: List[ChatMessage],
        functions: Optional[List[AIFunction]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        data = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        if functions:
            data["functions"] = [f.to_dict() for f in functions]

        try:
            async with self.client.stream("POST", "/chat/completions", json=data) as response:
                if response.status_code != 200:
                    error_detail = await response.text()
                    raise KaniException(f"Cerebras API returned status code {response.status_code}. Details: {error_detail}")

                async for line in response.aiter_lines():
                    if line.strip():
                        chunk = line.strip()
                        if chunk.startswith("data: "):
                            chunk = chunk[6:]
                        if chunk != "[DONE]":
                            try:
                                parsed_chunk = json.loads(chunk)
                                content = parsed_chunk["choices"][0]["delta"].get("content", "")
                                yield content
                            except json.JSONDecodeError:
                                continue

        except httpx.RequestError as e:
            raise KaniException(f"Error communicating with Cerebras API: {str(e)}")