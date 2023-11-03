CTransformers
=============
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
