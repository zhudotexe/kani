Extensions
==========
Extensions are 3rd-party packages that extend the core functionality available in kani with more domain-specific
tooling. With kani's philosophy of hackability, extensions can provide new engines, feature sets, prompting frameworks,
and even multimodality.

For 3rd party packages, we recommend using the ``kani.ext.*`` namespace (e.g. ``kani.ext.my_cool_extension``).

To get started with building a kani extension package, you can use the
`kani extension template repository <https://github.com/zhudotexe/kani-ext-template>`_!
This repo contains the right directory layout to use the ``kani.ext.*`` namespace and the PyPI packaging basics to
make your package available on pip.

Community Extensions
--------------------
If you've made a cool extension, add it to this list with a PR!

* **kani-multimodal-core**: Adds support for multimodal inputs: images, audio, and video.

  * ``pip install "kani[multimodal]"``
  * `GitHub (kani-multimodal-core) <https://github.com/zhudotexe/kani-multimodal-core>`_
  * `Docs (kani-multimodal-core) <https://kani-multimodal-core.readthedocs.io/en/latest/>`_

* **kani-ratelimits**: Adds a wrapper engine to enforce request-per-minute (RPM), token-per-minute (TPM), and/or
  max-concurrency ratelimits before making requests to an underlying engine.

  * ``pip install kani-ratelimits``
  * `GitHub (kani-ratelimits) <https://github.com/zhudotexe/kani-ratelimits>`_

* **kani-ext-vllm**: Adds support for loading models using vLLM with a local engine or managing a vLLM server process.
  The local vLLM engine is a drop-in replacement for a :class:`.HuggingEngine`.

  * ``pip install "kani[vllm]"``
  * `GitHub (kani-ext-vllm) <https://github.com/zhudotexe/kani-ext-vllm>`_

* **kani-ext-realtime**: Adds support for OpenAI's Realtime API and full-duplex voice models with Kani.

  * `GitHub (kani-ext-realtime) <https://github.com/zhudotexe/kani-ext-realtime>`_

Design Considerations
---------------------

Engine Extensions
^^^^^^^^^^^^^^^^^
When building an engine extension, you have a couple options. If your extension provides an implementation for a new
LLM, you can simply have your extension provide a new subclass of :class:`.BaseEngine` (see :doc:`../engines` for more
information on how to build an engine).

Engine extensions do not necessarily have to be an LLM, however; for example, you might provide specific
prompting frameworks as an engine that are compatible with any LLM of their own. In this case, we suggest having
your engine *wrap* another engine:

.. code-block:: python

    """An example showing how to wrap another kani engine."""

    from kani.engines import WrapperEngine

    # subclassing WrapperEngine automatically implements passthrough of untouched attributes
    # to the wrapped engine!
    class MyEngineWrapper(WrapperEngine):
        def prompt_len(self, messages, functions=None, **kwargs):
            # wrap the inner messages with the prompt framework...
            prompted_messages = [ChatMessage(...) for m in messages]
            return super().prompt_len(prompted_messages, functions, **kwargs)

        async def predict(self, messages, functions=None, **hyperparams):
            # wrap the messages with the prompt framework and pass it to the inner engine
            prompted_messages = [ChatMessage(...) for m in messages]
            prompted_completion = await super().predict(prompted_messages, ...)
            # unwrap the resulting message (if necessary) and store the metadata separately
            completion = self.unwrap(prompted_completion)
            return completion

The :class:`kani.engines.WrapperEngine` is a base class that automatically creates a constructor that takes in the
engine to wrap, and passes through any non-overriden attributes to the wrapped engine.

CLI Extensions
^^^^^^^^^^^^^^
An extension package can define CLI entrypoints for use with the Kani CLI. In order to do this, an extension package
must use the ``kani.ext.*`` namespace and define a ``CLI_PROVIDERS`` name in its ``__init__.py``. This variable should
be a 3-tuple: ``(name: str, aliases: list[str], entrypoint: (model_id) -> Engine)``.

For example, for the vLLM engine, which is provided through the ``kani.ext.vllm`` extension package, we define:

.. code-block:: python

    # kani/ext/vllm/__init__.py

    # ...
    CLI_PROVIDERS = [
        ("vllm", [], VLLMServerEngine),
    ]

The Kani CLI will automatically discover and provide CLI entrypoints defined in this way when called.
