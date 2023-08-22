Engines
=======
Engines are the means by which kani interact with language models. As you've seen, kani comes with a few engines
included:

.. include:: shared/engine_table.rst

In this section, we'll discuss how to implement your own engine to use any language model or API you can think of.

.. tip::

    Built an engine for a model kani doesn't support yet?
    kani is OSS and |:heart:| PRs with engine implementations for the latest models - see :doc:`contributing`.

Implementing an Engine
----------------------
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
^^^^^^^^^^^^^^^^
Engines also come with a set of optional methods/attributes to override that you can use to customize its behaviour
further. For example, engines often have to add a custom model-specific prompt in order to expose functions to
the underlying model, and kani needs to know about the extra tokens added by this prompt!

- :attr:`.BaseEngine.token_reserve`: if your engine needs to reserve tokens (e.g. for a one-time prompt template).
- :meth:`.BaseEngine.function_token_reserve`: specify how many tokens are needed to expose a set of functions to the
  model.
- :meth:`.BaseEngine.close`: if your engine needs to clean up resources during shutdown.

HTTP Client
-----------
If your language model backend exposes an HTTP API, you can create a subclass of :class:`.BaseClient` to interface with
it. Your engine should then create an instance of the new HTTP client and call it to make predictions.

Minimally, to use the HTTP client, your subclass should set the ``SERVICE_BASE`` class variable.

.. seealso::

    The source code of the :class:`.OpenAIClient`, which uses the HTTP client.

.. autoclass:: kani.engines.httpclient.BaseClient
    :noindex:
    :members:

HuggingFace
-----------
If your language model backend is available on HuggingFace or is compatible with ``transformers``'
``AutoModelForCausalLM`` interface, kani includes a base engine that implements a prediction pipeline.

Instead of having to implement the prediction logic, all you have to do is subclass :class:`.HuggingEngine` and
implement :meth:`~.HuggingEngine.build_prompt` and :meth:`~.BaseEngine.message_len`.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you're running your model locally, you might run into issues because large language models are, well, *large*!
Unless you pay for a massive compute cluster (|:money_with_wings:|) or have access to one at your institution, you
might not be able to fit models with billions of params on your GPU. That's where model quantization comes into play.

    Using FP4 quantization you can expect to reduce up to 8x the model size compared to its native full precision
    version.

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
    :emphasize-lines: 4-7

    engine = LlamaEngine(
        use_auth_token=True,
        strict=True,
        model_load_kwargs={
            "device_map": "auto",
            "load_in_4bit": True,
        },
    )


**Memory Usage Comparison**

This table shows the effect of enabling fp4 quantization on GPU memory usage and inference speed on ``Llama-2-7b-chat``.

These numbers represent the average of three runs on a consumer RTX 4070ti (12GB memory) with greedy sampling.

+--------------+----------------------+----------------------------------------+
| fp4 Enabled? | Memory Usage         | Inference Time (per token)             |
+==============+======================+========================================+
| No           | 26.6GB               | 1215.6 ms                              |
+--------------+----------------------+----------------------------------------+
| Yes          | 5.0GB (5.32x less)   | 23.6 ms (51.5x speedup\ [#shared]_)    |
+--------------+----------------------+----------------------------------------+

.. [#shared] Since the memory usage without fp4 enabled is larger than the VRAM size of my GPU, some weights were stored
    in shared memory. This likely led to much slower inference compared to storing all weights on a GPU.

CTransformers
-------------
If your language model backend is available with GGML, kani includes a base engine that implements 
a prediction pipeline.

Instead of having to implement the prediction logic, all you have to do is subclass :class:`.CTransformersEngine` and
implement :meth:`~.CTransformersEngine.build_prompt` and :meth:`~.BaseEngine.message_len`.

.. seealso::

    The source code of the :class:`.LlamaCTransformersEngine`, which uses the CTransformersEngine.

.. autoclass:: kani.engines.ctransformers.base.CTransformersEngine
    :noindex:

    .. automethod:: kani.engines.ctransformers.base.CTransformersEngine.build_prompt
        :noindex:

    .. automethod:: kani.engines.ctransformers.base.CTransformersEngine.message_len
        :noindex:
