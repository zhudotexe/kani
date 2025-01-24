HuggingFace
===========
If your language model backend is available on HuggingFace or is compatible with ``transformers``'
``AutoModelForCausalLM`` interface, kani includes a base engine that implements a prediction pipeline.

.. important::

    .. versionadded:: 1.2.0
        For most models that use a chat template, you won't need to create a new engine class - kani will automatically
        use a `Chat Template <https://huggingface.co/docs/transformers/main/en/chat_templating>`_ if a model has one
        included.

        This means you can safely ignore this section of the documentation for most use cases! Just use:

        .. code-block:: python

            from kani.engines.huggingface import HuggingEngine
            engine = HuggingEngine(model_id="your-org/your-model-id")

.. versionadded:: 1.0.0
    For more control over the prompting of a chat model, you can pass a :class:`.PromptPipeline` to
    the :class:`.HuggingEngine`.

If you do create a new engine, instead of having to implement the prediction logic, all you have to do is subclass
:class:`.HuggingEngine` and implement :meth:`~.HuggingEngine.build_prompt` and :meth:`~.BaseEngine.message_len`.

.. seealso::

    The source code of the :class:`.LlamaEngine`, which uses the HuggingEngine.

.. autoclass:: kani.engines.huggingface.base.HuggingEngine
    :noindex:

    .. automethod:: kani.engines.huggingface.base.HuggingEngine.build_prompt
        :noindex:

    .. automethod:: kani.engines.huggingface.base.HuggingEngine.message_len
        :noindex:

.. _4b_quant:

4-bit Quantization (|:hugging:|)
--------------------------------
If you're running your model locally, you might run into issues because large language models are, well, *large*!
Unless you pay for a massive compute cluster (|:money_with_wings:|) or have access to one at your institution, you
might not be able to fit models with billions of params on your GPU. That's where model quantization comes into play.

.. tip::

    Thanks to the hard work of the LLM community, many models on Hugging Face also have quantized versions available
    in the GGUF format. GGUF is the format for ``llama.cpp``, a low-level optimized LLM runtime. Unlike the name
    suggests, it supports many more models than LLaMA. If your model has a GGUF version available, consider using the
    :class:`.LlamaCppEngine` instead of the ``HuggingEngine`` to load a pre-quantized version.

In this section, we'll show how to load HuggingFace models in FP4.

.. seealso::

    We're mostly going to follow the HuggingFace documentation found here:
    https://huggingface.co/docs/transformers/perf_infer_gpu_one

**Install Dependencies**

First, you'll need to install kani with the ``huggingface`` extra (and any other extras necessary for your engine;
we'll use LLaMA v2 in this example, so you'll want ``pip install 'kani[huggingface,llama]'``\ .)

After that, you'll need to install ``bitsandbytes`` and ``accelerate``:

.. code-block:: console

    $ pip install bitsandbytes>=0.39.0 accelerate

.. caution:: The ``bitsandbytes`` library is currently only UNIX-compatible.

**Set Load Arguments**

Then, you'll need to set the ``model_load_kwargs`` when initializing your model, and use the engine as normal! This
example shows the :class:`.LlamaEngine`, but the same arguments should apply to any subclass of the
:class:`.HuggingEngine`.

.. code-block:: python

    from transformers import BitsAndBytesConfig
    from kani.engines.huggingface.llama2 import LlamaEngine

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    engine = LlamaEngine(
        use_auth_token=True,
        model_load_kwargs={
            "device_map": "auto",
            "quantization_config": quantization_config,
        },
    )
