Implementing an Engine
======================

.. important::
    Looking to use a model available through HuggingFace Transformers? Before implementing your own model class, try
    using :class:`.HuggingEngine`! If your model has a Chat Template available, Kani will automatically use the correct
    prompt format.

    .. code-block:: python

            from kani.engines.huggingface import HuggingEngine
            engine = HuggingEngine(model_id="your-org/your-model-id")

To create your own engine, all you have to do is subclass :class:`.BaseEngine`:

.. autoclass:: kani.engines.base.BaseEngine
    :noindex:
    :members:

A new engine must implement at least the two abstract methods and set the abstract attribute:

- :meth:`.BaseEngine.message_len` takes a single :class:`.ChatMessage` and returns the length of that message,
  in tokens.
- :meth:`.BaseEngine.predict` takes a list of :class:`.ChatMessage` and :class:`.AIFunction` and returns a
  new :class:`.BaseCompletion`.
- :attr:`.BaseEngine.max_context_size` specifies the model's token context size.

With just these three implementations, an engine will be fully functional!

kani comes with a couple additional bases and utilities to help you build engines for models on HuggingFace or with
an available HTTP API.

Optional Methods
----------------
Engines also come with a set of optional methods/attributes to override that you can use to customize its behaviour
further. For example, engines often have to add a custom model-specific prompt in order to expose functions to
the underlying model, and kani needs to know about the extra tokens added by this prompt!

- :attr:`.BaseEngine.token_reserve`: if your engine needs to reserve tokens (e.g. for a one-time prompt template).
- :meth:`.BaseEngine.function_token_reserve`: specify how many tokens are needed to expose a set of functions to the
  model.
- :meth:`.BaseEngine.close`: if your engine needs to clean up resources during shutdown.

Adding Function Calling
-----------------------

.. important::
    Already have a way to build function calling prompts but just need a way to parse the outputs? Check out the list
    of :ref:`tool-parsers`.

If you're writing an engine for a model with function calling, there are a couple additional steps you need to take.

Generally, to use function calling, you need to do the following:

1. Tell the model what functions it has available to it
    a. Optional - tell the model what format to output to request calling a function (if the model is not already
       fine-tuned to do so)
2. Parse the model's requests to call functions from its text generations

To tell the model what functions it has available, you'll need to somehow prompt the model.
You'll need to implement two methods: :meth:`.BaseEngine.predict` and :meth:`.BaseEngine.function_token_reserve`.

:meth:`.BaseEngine.predict` takes in a list of available :class:`.AIFunction`\ s as an argument, which you should use to
build such a prompt. :meth:`.BaseEngine.function_token_reserve` tells kani how many tokens that prompt takes, so the
context window management can ensure it never sends too many tokens.

You'll also need to add previous function calls into the prompt (e.g. in the few-shot function calling example).
When you're building the prompt, you'll need to iterate over :attr:`.ChatMessage.tool_calls` if it exists, and add
your model's appropriate function calling prompt.

To parse the model's requests to call a function, you also do this in :meth:`.BaseEngine.predict`. After generating the
model's completion (usually a string, or a list of token IDs that decodes into a string), separate the model's
conversational content from the structured function call:

.. image:: /_static/function-calling-parsing.png
    :align: center

Finally, return a :class:`.Completion` with the ``.message`` attribute set to a :class:`.ChatMessage` with the
appropriate :attr:`.ChatMessage.content` and :attr:`.ChatMessage.tool_calls`.

.. note::
    See :ref:`functioncall_v_toolcall` for more information about ToolCalls vs FunctionCalls.
