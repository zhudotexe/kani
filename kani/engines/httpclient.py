import abc
import asyncio
import logging

from ..exceptions import HTTPException, HTTPStatusException, HTTPTimeout, MissingModelDependencies

try:
    import aiohttp
except ImportError:
    raise MissingModelDependencies(
        f"The BaseClient has been deprecated as of v1.0.0. We recommend using `httpx` for HTTP requests. To continue"
        f" using the BaseClient (not recommended), use `pip install aiohttp`."
    )


class BaseClient(abc.ABC):
    """aiohttp-based HTTP client to help implement HTTP-based engines.

    .. deprecated:: 1.0.0
        We recommend using `httpx.AsyncClient <https://www.python-httpx.org/async/>`_ instead. This aiohttp-based client
        will be removed in a future version.
    """

    SERVICE_BASE: str
    """The base route of the HTTP API."""

    logger: logging.Logger

    def __init__(self, http: aiohttp.ClientSession = None):
        """
        :param http: The :class:`aiohttp.ClientSession` to use; if not provided, creates a new session.
        """
        self.http = http

    def __init_subclass__(cls, **kwargs):
        # to initialize the logger with the right name, set it when the subclass is initialized
        # this prevents all loggers logging as kani.engines.httpclient
        cls.logger = logging.getLogger(cls.__module__)

    async def request(self, method: str, route: str, **kwargs) -> aiohttp.ClientResponse:
        """Makes an HTTP request to the given route (relative to the base route).

        :param method: The HTTP method to use (e.g. 'GET', 'POST').
        :param route: The route to make the request to (relative to the ``SERVICE_BASE``).
        :raises HTTPStatusException: The request returned a non-2xx response.
        :raises HTTPTimeout: The request timed out.
        :raises HTTPException: The response could not be deserialized.
        """
        if self.http is None:
            self.http = aiohttp.ClientSession()
        try:
            async with self.http.request(method, f"{self.SERVICE_BASE}{route}", **kwargs) as resp:
                self.logger.debug(f"{method} {self.SERVICE_BASE}{route} returned {resp.status}")
                if not 199 < resp.status < 300:
                    data = await resp.text()
                    self.logger.warning(
                        f"{method} {self.SERVICE_BASE}{route} returned {resp.status} {resp.reason}\n{data}"
                    )
                    raise HTTPStatusException(resp, f"Request returned an error: {resp.status}: {resp.reason}")
                # hydrate the response body into cache; this allows reading the response after exiting the context
                # https://stackoverflow.com/questions/68693855/aiohttp-getting-response-object-out-of-context-manager
                await resp.read()
        except asyncio.TimeoutError as e:
            self.logger.warning(f"Request timeout: {method} {self.SERVICE_BASE}{route}")
            raise HTTPTimeout() from e
        return resp

    async def _deserialize_response(self, method: str, route: str, resp: aiohttp.ClientResponse):
        try:
            data = await resp.json()
            self.logger.debug(data)
            return data
        except (aiohttp.ContentTypeError, ValueError, TypeError) as e:
            data = await resp.text()
            self.logger.warning(f"{method} {self.SERVICE_BASE}{route} response could not be deserialized:\n{data}")
            raise HTTPException(f"Could not deserialize response: {data}") from e

    async def get(self, route: str, **kwargs):
        """Convenience method; equivalent to ``self.request("GET", route, **kwargs).json()``."""
        resp = await self.request("GET", route, **kwargs)
        return await self._deserialize_response("GET", route, resp)

    async def post(self, route: str, **kwargs):
        """Convenience method; equivalent to ``self.request("POST", route, **kwargs).json()``."""
        resp = await self.request("POST", route, **kwargs)
        return await self._deserialize_response("POST", route, resp)

    async def close(self):
        """Close the underlying aiohttp session."""
        if self.http is None:
            return
        await self.http.close()
        self.http = None  # this allows us to reuse the client after it has been closed
