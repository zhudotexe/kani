import abc
import logging

import aiohttp

from chatterbox.exceptions import HTTPStatusException, HTTPException, HTTPTimeout


class BaseClient(abc.ABC):
    SERVICE_BASE: str = ...
    logger: logging.Logger

    def __init__(self, http: aiohttp.ClientSession):
        self.http = http

    def __init_subclass__(cls, **kwargs):
        # to initialize the logger with the right name, set it when the subclass is initialized
        # this prevents all loggers logging as chatterbox.engines.httpclient
        cls.logger = logging.getLogger(cls.__module__)

    async def request(self, method: str, route: str, response_as_text=False, **kwargs):
        try:
            async with self.http.request(method, f"{self.SERVICE_BASE}{route}", **kwargs) as resp:
                self.logger.debug(f"{method} {self.SERVICE_BASE}{route} returned {resp.status}")
                if not 199 < resp.status < 300:
                    data = await resp.text()
                    self.logger.warning(
                        f"{method} {self.SERVICE_BASE}{route} returned {resp.status} {resp.reason}\n{data}"
                    )
                    raise HTTPStatusException(resp.status, f"Request returned an error: {resp.status}: {resp.reason}")
                try:
                    if not response_as_text:
                        data = await resp.json()
                    else:
                        data = await resp.text()
                    self.logger.debug(data)
                except (aiohttp.ContentTypeError, ValueError, TypeError) as e:
                    data = await resp.text()
                    self.logger.warning(
                        f"{method} {self.SERVICE_BASE}{route} response could not be deserialized:\n{data}"
                    )
                    raise HTTPException(f"Could not deserialize response: {data}") from e
        except aiohttp.ServerTimeoutError as e:
            self.logger.warning(f"Request timeout: {method} {self.SERVICE_BASE}{route}")
            raise HTTPTimeout() from e
        return data

    async def get(self, route: str, **kwargs):
        return await self.request("GET", route, **kwargs)

    async def post(self, route: str, **kwargs):
        return await self.request("POST", route, **kwargs)

    async def close(self):
        await self.http.close()
