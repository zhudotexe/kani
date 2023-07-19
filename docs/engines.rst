Engines
=======
Engines are the means by which kani interact with language models. As you've seen, kani comes with a few engines
included:

.. include:: shared/engine_table.rst

In this section, we'll discuss how to implement your own engine to use any language model or API you can think of.

.. tip::

    Built an engine for a model kani doesn't support yet?
    kani is OSS and |:heart:| PRs with engine implementations for the latest models.

Implementing an Engine
----------------------
To create your own engine, all you have to do is subclass :class:`.BaseEngine`:

.. autoclass:: kani.engines.base.BaseEngine
    :noindex:
    :members:

A new engine must implement at least the two abstract methods and set the abstract attribute:

- :meth:`.BaseEngine.message_len` takes a single :class:`.ChatMessage` and returns the length of that message,
  in tokens.
- :meth:`.BaseEngine.predict` takes a list of :class:`.ChatMessage` and :class:`.AIFunction` and returns a
  new :class:`.BaseCompletion`.
- :attr:`.BaseEngine.max_context_size` specifies the model's token context size.

Optionally, you can also implement :meth:`.BaseEngine.close` if your engine needs to clean up resources, and
:attr:`.BaseEngine.token_reserve` if your engine needs to reserve tokens (e.g. for a one-time prompt template).

kani comes with a couple additional bases and utilities to help you build engines for models on Hugging Face or with
an available HTTP API.

HTTP Client
-----------
If your language model backend exposes an HTTP API, you can create a subclass of :class:`.BaseClient` to interface with
it. Your engine should then create an instance of the new HTTP client and call it to make predictions.

.. seealso::

    The source code of the :class:`.OpenAIClient`, which uses the HTTP client.

.. autoclass:: kani.engines.httpclient.BaseClient
    :noindex:
    :members:

Hugging Face
------------
If your language model backend is available on Hugging Face or is compatible with ``transformers``'
``AutoModelForCausalLM`` interface, kani includes a base engine that implements a prediction pipeline.

Instead of having to implement the prediction logic, all you have to do is subclass :class:`.HuggingEngine` and
implement :meth:`~.HuggingEngine.build_prompt` and :meth:`~.BaseEngine.message_len`.

.. seealso::

    The source code of the :class:`.LlamaEngine`, which uses the HuggingEngine.

.. autoclass:: kani.engines.huggingface.base.HuggingEngine
    :noindex:

    .. automethod:: kani.engines.huggingface.base.HuggingEngine.build_prompt
        :noindex:

    .. automethod:: kani.engines.huggingface.base.HuggingEngine.message_len
        :noindex:

Contributing to kani
--------------------
If you've implemented an engine for a new model, we'd love to include it in kani!

To make a PR to kani with a new engine implementation, follow these steps:

1. Add your engine implementation to ``/kani/engines``.
2. If your engine requires extra dependencies, add them as extras to ``requirements.txt`` and ``pyproject.toml``.
3. Add your engine to the docs in ``/docs/shared/engine_table.rst`` and ``/docs/engine_reference.rst``.
4. Open a PR!
