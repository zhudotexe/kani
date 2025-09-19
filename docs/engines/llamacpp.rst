LlamaCppEngine
==============
If your language model backend is available with GGUF, kani includes a base engine that implements
a prediction pipeline.

**TL;DR**

.. code-block:: python

    from kani.engines.huggingface import ChatTemplatePromptPipeline
    from kani.engines.llamacpp import LlamaCppEngine
    pipeline = ChatTemplatePromptPipeline.from_pretrained("org-id/base-model-id")
    engine = LlamaCppEngine(repo_id="org-id/quant-model-id", filename="*.your-quant-type.gguf", prompt_pipeline=pipeline)

.. important::

    .. versionadded:: 1.4.0
        For most models that use a chat template, you won't need to create a new engine class - kani will automatically
        use a `Chat Template <https://huggingface.co/docs/transformers/main/en/chat_templating>`_ if a model has one
        included.

        This means you can safely ignore this section of the documentation for most use cases! Just use:

        .. code-block:: python

            from kani.engines.llamacpp import LlamaCppEngine
            engine = LlamaCppEngine(repo_id="your-org/your-model-id", filename="*Q4_K_M.gguf")

kani uses `llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_ for binding to the llama.cpp runtime.

.. versionadded:: 1.0.0
    For most models that use a chat format, you won't even need to create a new engine class - instead, you can pass
    a :class:`.PromptPipeline` to the :class:`.LlamaCppEngine`.

If you do create a new engine, instead of having to implement the prediction logic, all you have to do is subclass
:class:`.LlamaCppEngine` and implement :meth:`~.LlamaCppEngine.build_prompt` and :meth:`~.LlamaCppEngine.message_len`.

.. autoclass:: kani.engines.llamacpp.LlamaCppEngine
    :noindex:

    .. automethod:: kani.engines.llamacpp.LlamaCppEngine.build_prompt
        :noindex:

    .. automethod:: kani.engines.llamacpp.LlamaCppEngine.message_len
        :noindex:
