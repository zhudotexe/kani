Engine Reference
================

.. include:: shared/engine_table.rst

OpenAI
------
.. autoclass:: kani.engines.openai.OpenAIEngine

Anthropic
---------
.. autoclass:: kani.engines.anthropic.AnthropicEngine

.. autoclass:: kani.engines.anthropic.AnthropicPDFFilePart

Hugging Face
------------
.. autoclass:: kani.engines.huggingface.HuggingEngine
    :members:

.. autoclass:: kani.engines.huggingface.llama2.LlamaEngine
    :members:

.. autoclass:: kani.engines.huggingface.cohere.CommandREngine
    :members:

.. autoclass:: kani.engines.huggingface.vicuna.VicunaEngine
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
