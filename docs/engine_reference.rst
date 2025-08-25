Engine Reference
================

.. include:: shared/engine_table.rst

OpenAI
------
.. autoclass:: kani.engines.openai.OpenAIEngine

Anthropic
---------
.. autoclass:: kani.engines.anthropic.AnthropicEngine

Google AI
---------
.. autoclass:: kani.engines.google.GoogleAIEngine

Hugging Face
------------
.. autoclass:: kani.engines.huggingface.HuggingEngine
    :members:

llama.cpp
---------
.. autoclass:: kani.engines.llamacpp.LlamaCppEngine
    :members:

Base
----
.. autoclass:: kani.engines.BaseEngine
    :members:

.. autoclass:: kani.engines.Completion
    :members:

.. autoclass:: kani.engines.WrapperEngine

    .. autoattribute:: engine

.. autoclass:: kani.engines.base.BaseCompletion
    :members:

.. autoclass:: kani.engines.httpclient.BaseClient
    :members:
