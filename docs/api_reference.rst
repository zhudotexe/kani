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

AI Function
-----------
.. autofunction:: kani.ai_function

.. autoclass:: kani.AIFunction
    :members:

.. autoclass:: kani.AIParam
    :members:

Common MessageParts
-------------------
.. autoclass:: kani.parts.ReasoningPart
    :members:
    :exclude-members: model_config, model_fields, model_computed_fields
    :class-doc-from: class

Exceptions
----------
.. automodule:: kani.exceptions
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

Model-Specific Parsers
^^^^^^^^^^^^^^^^^^^^^^
Model parsers are used when you have an LLM's text output, which may contain tool calls or other interleaved content
in their raw format (e.g., reasoning output).
They translate the raw text format into Kani's tool calling specification and MessageParts.

Tool parsers are :class:`.WrapperEngine`\ s -- this means to use them, you should **wrap** the text-only engine (e.g.,
a :class:`.HuggingEngine`) like so:

.. code-block:: python

    from kani.engines.huggingface import HuggingEngine
    from kani.tool_parsers import GPTOSSParser

    model = HuggingEngine("openai/gpt-oss-20b")
    engine = GPTOSSParser(model)

.. autoclass:: kani.model_specific.BaseToolCallParser
    :members:

.. autoclass:: kani.model_specific.gpt_oss.GPTOSSParser

.. autoclass:: kani.model_specific.json.NaiveJSONToolCallParser

.. autoclass:: kani.model_specific.mistral.MistralToolCallParser

.. autoclass:: kani.model_specific.deepseek.DeepSeekR1ToolCallParser
