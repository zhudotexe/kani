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
.. autoclass:: kani.engines.BaseEngine
    :members:

.. autoclass:: kani.engines.base.BaseCompletion
    :members:

OpenAI
^^^^^^
.. autoclass:: kani.engines.OpenAIEngine

.. autoclass:: kani.engines.openai.client.OpenAIClient
    :members:

Utilities
---------
.. autofunction:: kani.chat_in_terminal
