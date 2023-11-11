import asyncio
import warnings
from typing import Literal, overload

import aiohttp
import pydantic

from .models import (
    ChatCompletion,
    Completion,
    FunctionSpec,
    OpenAIChatMessage,
    ResponseFormat,
    SpecificFunctionCall,
    ToolChoice,
    ToolSpec,
)
from ..httpclient import BaseClient, HTTPException, HTTPStatusException, HTTPTimeout


class OpenAIClient(BaseClient):
    """Simple HTTP client to interface with the OpenAI API.

    This client includes exponential backoff on retryable errors.
    """

    def __init__(
        self,
        api_key: str,
        http: aiohttp.ClientSession = None,
        organization: str = None,
        retry: int = 5,
        api_base: str = "https://api.openai.com/v1",
        headers: dict = None,
    ):
        """
        :param api_key: The OpenAI API key to use.
        :param http: An aiohttp session to use (creates one by default).
        :param organization: The OpenAI org ID to bill requests to (default based on api key).
        :param retry: How many times to retry a failed request (default 5).
        :param api_base: The base URL of the API to use (default OpenAI).
        :param headers: Any additional headers to send with each request.
        """
        if headers is None:
            headers = {}
        super().__init__(http)
        self.api_key = api_key
        self.organization = organization
        self.retry = retry
        self.SERVICE_BASE = api_base
        self.headers = headers

    async def request(self, method: str, route: str, headers=None, retry=None, **kwargs):
        if headers is None:
            headers = self.headers.copy()
        # set up auth headers
        if "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        # make the request
        retry = retry if retry is not None else self.retry
        for i in range(retry):
            try:
                return await super().request(method, route, headers=headers, **kwargs)
            except (HTTPStatusException, HTTPTimeout) as e:
                # raise if out of retries
                if (i + 1) >= retry:
                    raise

                # handle http status exceptions
                # non-retryable error (see https://platform.openai.com/docs/guides/error-codes/api-errors)
                if isinstance(e, HTTPStatusException) and e.status_code in (400, 401, 403, 404):
                    raise

                # otherwise, wait and try again
                retry_sec = 2**i
                self.logger.warning(f"OpenAI returned {e}, retrying in {retry_sec} sec...")
                await asyncio.sleep(retry_sec)
        raise RuntimeError(
            "Ran out of retries but no error encountered. This should never happen; please report an issue on GitHub!"
        )

    # ==== completions ====
    @overload
    async def create_completion(
        self,
        model: str,
        *,
        prompt: str = "<|endoftext|>",
        suffix: str = None,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        logprobs: int = None,
        echo: bool = False,
        seed: int | None = None,
        stop: str | list[str] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        best_of: int = 1,
        logit_bias: dict = None,
        user: str = None,
    ) -> Completion: ...

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
        messages: list[OpenAIChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | Literal["auto"] | Literal["none"] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        response_format: ResponseFormat | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        max_tokens: int | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: dict | None = None,
        user: str | None = None,
        # deprecated
        functions: list[FunctionSpec] | None = None,
        function_call: SpecificFunctionCall | Literal["auto"] | Literal["none"] | None = None,
    ) -> ChatCompletion: ...

    async def create_chat_completion(
        self,
        model: str,
        messages: list[OpenAIChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        """Create a chat completion.

        See https://platform.openai.com/docs/api-reference/chat/create.
        """
        # transform pydantic models
        if tools:
            kwargs["tools"] = [t.model_dump(exclude_unset=True) for t in tools]
        if "tool_choice" in kwargs and isinstance(kwargs["tool_choice"], SpecificFunctionCall):
            kwargs["tool_choice"] = kwargs["tool_choice"].model_dump(exclude_unset=True)
        # deprecated function calling
        if "functions" in kwargs:
            warnings.warn("The functions parameter is deprecated. Use tools instead.", DeprecationWarning)
            kwargs["functions"] = [f.model_dump(exclude_unset=True) for f in kwargs["functions"]]
        if "function_call" in kwargs and isinstance(kwargs["function_call"], SpecificFunctionCall):
            warnings.warn("The function_call parameter is deprecated. Use tool_choice instead.", DeprecationWarning)
            kwargs["function_call"] = kwargs["function_call"].model_dump(exclude_unset=True)
        # call API
        data = await self.post(
            "/chat/completions",
            json={
                "model": model,
                # this needs to be exclude_defaults since content can be None but is required regardless
                "messages": [cm.model_dump(exclude_defaults=True, mode="json") for cm in messages],
                **kwargs,
            },
        )
        try:
            return ChatCompletion.model_validate(data)
        except pydantic.ValidationError:
            self.logger.exception(f"Failed to deserialize OpenAI response: {data}")
            raise HTTPException(f"Could not deserialize response: {data}")
