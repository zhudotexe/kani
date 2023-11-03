HuggingFace
===========
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
--------------------------------
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
