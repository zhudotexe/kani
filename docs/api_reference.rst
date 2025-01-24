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
    :exclude-members: model_config, model_fields, model_computed_fields
    :class-doc-from: class

.. autoclass:: kani.ToolCall
    :members:
    :exclude-members: model_config, model_fields, model_computed_fields
    :class-doc-from: class

.. autoclass:: kani.MessagePart
    :members:
    :exclude-members: model_config, model_fields, model_computed_fields
    :class-doc-from: class

    .. automethod:: __str__

.. autoclass:: kani.ChatMessage
    :members:
    :exclude-members: model_config, model_fields, model_computed_fields
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

Streaming
---------
.. autoclass:: kani.streaming.StreamManager
    :members:

    .. automethod:: __await__

    .. automethod:: __aiter__

Prompting
---------
.. automodule:: kani.prompts

.. autoclass:: kani.PromptPipeline
    :members:

    .. automethod:: __call__

.. autoclass:: kani.prompts.PipelineStep
    :members:

.. autoclass:: kani.prompts.ApplyContext
    :members:

Internals
---------
.. autoclass:: kani.FunctionCallResult
    :members:

.. autoclass:: kani.ExceptionHandleResult
    :members:

Engines
-------
See :doc:`engine_reference`.

Utilities
---------
.. autofunction:: kani.chat_in_terminal

.. autofunction:: kani.chat_in_terminal_async

.. autofunction:: kani.format_width

.. autofunction:: kani.format_stream

.. autofunction:: kani.print_width

.. autofunction:: kani.print_stream

Message Formatters
^^^^^^^^^^^^^^^^^^
.. automodule:: kani.utils.message_formatters
    :members:

.. _tool-parsers:

Tool Parsers
^^^^^^^^^^^^
Tool parsers are used when you have an LLM's text output, which may contain tool calls in their raw format (e.g., JSON).
They translate the raw text format into Kani's tool calling specification.

.. autoclass:: kani.tool_parsers.BaseToolCallParser
    :members:

.. autoclass:: kani.tool_parsers.NaiveJSONToolCallParser

.. autoclass:: kani.tool_parsers.MistralToolCallParser
