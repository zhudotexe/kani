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
If you've made a cool extension, add it to this table with a PR!

+-------------+----------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| Name        | Description                                                                | Links                                                                                                        |
+=============+============================================================================+==============================================================================================================+
| kani-vision | Adds support for multimodal vision-language models, like GPT-4V and LLaVA. | `GitHub <https://github.com/zhudotexe/kani-vision>`_ `Docs <https://kani-vision.readthedocs.io/en/latest/>`_ |
+-------------+----------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------+

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

    class MyEngineWrapper(BaseEngine):
        def __init__(self, inner_engine: BaseEngine):
            self.inner_engine = inner_engine
            self.max_context_size = inner_engine.max_context_size

        def message_len(self, message):
            # wrap the inner message with the prompt framework...
            prompted_message = ChatMessage(...)
            return self.inner_engine.message_len(prompted_message)

        async def predict(self, messages, functions=None, **hyperparams):
            # wrap the messages with the prompt framework and pass it to the inner engine
            prompted_completion = await self.inner_engine.predict(prompted_messages, ...)
            # unwrap the resulting message (if necessary) and store the metadata separately
            completion = self.unwrap(prompted_completion)
            return completion
