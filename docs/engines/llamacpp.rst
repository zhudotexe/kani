llama.cpp
=========
If your language model backend is available with GGUF, kani includes a base engine that implements
a prediction pipeline.

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
