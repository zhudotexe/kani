HTTP Client
===========
If your language model backend exposes an HTTP API, you can create a subclass of :class:`.BaseClient` to interface with
it. Your engine should then create an instance of the new HTTP client and call it to make predictions.

Minimally, to use the HTTP client, your subclass should set the ``SERVICE_BASE`` class variable.

.. seealso::

    The source code of the :class:`.OpenAIClient`, which uses the HTTP client.

.. autoclass:: kani.engines.httpclient.BaseClient
    :noindex:
    :members:
