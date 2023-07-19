API Reference
=============

Kani
----
.. autoclass:: kani.Kani
    :members:

Common Models
-------------
.. autoclass:: kani.models.ChatRole
    :members:

.. autoclass:: kani.models.FunctionCall
    :members:
    :class-doc-from: class

.. autoclass:: kani.models.ChatMessage
    :members:
    :class-doc-from: class

Exceptions
----------
.. automodule:: kani.exceptions
    :members:

AI Function
-----------
.. autofunction:: kani.ai_function

.. autoclass:: kani.ai_function.AIFunction
    :members:

.. autoclass:: kani.ai_function.AIParam
    :members:

Engines
-------
.. autoclass:: kani.engines.base.BaseEngine
    :members:

.. autoclass:: kani.engines.base.BaseCompletion
    :members:

.. autoclass:: kani.engines.base.Completion
    :members:

.. autoclass:: kani.engines.httpclient.BaseClient
    :members:

OpenAI
^^^^^^
.. autoclass:: kani.engines.openai.OpenAIEngine

.. autoclass:: kani.engines.openai.client.OpenAIClient
    :members:

HuggingFace
^^^^^^^^^^^
.. autoclass:: kani.engines.huggingface.base.HuggingEngine
    :members:

.. autoclass:: kani.engines.huggingface.vicuna.VicunaEngine
    :members:

Utilities
---------
.. autofunction:: kani.chat_in_terminal
