HTTP Client
===========

.. danger::
    The aiohttp-based BaseClient has been deprecated in v1.0.0 and will be removed in a future version.
    We recommend using `httpx <https://www.python-httpx.org/>`_ to make HTTP requests instead.

    We have removed the top-level library dependency on ``aiohttp`` - existing code using the BaseClient will require
    manual installation of the ``aiohttp`` package.

If your language model backend exposes an HTTP API, you can create a subclass of :class:`.BaseClient` to interface with
it. Your engine should then create an instance of the new HTTP client and call it to make predictions.

Minimally, to use the HTTP client, your subclass should set the ``SERVICE_BASE`` class variable.

.. autoclass:: kani.engines.httpclient.BaseClient
    :noindex:
    :members:
