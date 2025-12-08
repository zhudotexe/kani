Engine Reference
================

.. include:: shared/engine_table.rst

OpenAI
------
.. autoclass:: kani.engines.openai.OpenAIEngine

Anthropic
---------
.. autoclass:: kani.engines.anthropic.AnthropicEngine

.. autoclass:: kani.engines.anthropic.AnthropicUnknownPart
    :class-doc-from: class
    :members:
    :exclude-members: model_config, model_fields, model_computed_fields

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

vLLM
----
See the ``kani-ext-vllm`` documentation at https://github.com/zhudotexe/kani-ext-vllm.

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
