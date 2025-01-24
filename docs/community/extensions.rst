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

* **kani-ratelimits**: Adds a wrapper engine to enforce request-per-minute (RPM), token-per-minute (TPM), and/or
  max-concurrency ratelimits before making requests to an underlying engine.

  * `GitHub (kani-ratelimits) <https://github.com/zhudotexe/kani-ratelimits>`_

* **kani-ext-vllm**: Adds support for loading models using vLLM. Supports chat templates - should be mostly a drop-in
  replacement for a :class:`.HuggingEngine`.

  * `GitHub (kani-ext-vllm) <https://github.com/zhudotexe/kani-ext-vllm>`_

* **kani-vision**: Adds support for multimodal vision-language models, like GPT-4V and LLaVA.

  * `GitHub (kani-vision) <https://github.com/zhudotexe/kani-vision>`_
  * `Docs (kani-vision) <https://kani-vision.readthedocs.io/en/latest/>`_

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

    from kani.engines import BaseEngine, WrapperEngine

    # subclassing WrapperEngine automatically implements passthrough of untouched attributes
    # to the wrapped engine!
    class MyEngineWrapper(WrapperEngine):
        def message_len(self, message):
            # wrap the inner message with the prompt framework...
            prompted_message = ChatMessage(...)
            return super().message_len(prompted_message)

        async def predict(self, messages, functions=None, **hyperparams):
            # wrap the messages with the prompt framework and pass it to the inner engine
            prompted_completion = await super().predict(prompted_messages, ...)
            # unwrap the resulting message (if necessary) and store the metadata separately
            completion = self.unwrap(prompted_completion)
            return completion

The :class:`kani.engines.WrapperEngine` is a base class that automatically creates a constructor that takes in the
engine to wrap, and passes through any non-overriden attributes to the wrapped engine.
