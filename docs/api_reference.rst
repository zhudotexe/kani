API Reference
=============

Kani
----
.. autoclass:: kani.Kani
    :members:

Common Models
-------------
.. autoclass:: kani.ChatRole
    :members:

.. autoclass:: kani.FunctionCall
    :members:
    :exclude-members: model_config, model_fields
    :class-doc-from: class

.. autoclass:: kani.ChatMessage
    :members:
    :exclude-members: model_config, model_fields
    :class-doc-from: class

Exceptions
----------
.. automodule:: kani.exceptions
    :members:

AI Function
-----------
.. autofunction:: kani.ai_function

.. autoclass:: kani.AIFunction
    :members:

.. autoclass:: kani.AIParam
    :members:

Engines
-------
See :doc:`engine_reference`.

Utilities
---------
.. autofunction:: kani.chat_in_terminal

.. autofunction:: kani.chat_in_terminal_async
