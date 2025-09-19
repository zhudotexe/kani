OpenAIEngine
============
The :class:`.OpenAIEngine` is used to make requests to the OpenAI API.

**TL;DR**

.. code-block:: python

    # see https://platform.openai.com/docs/models for a list of model IDs
    from kani.engines.openai import OpenAIEngine
    engine = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5-nano")

Reference
---------

.. autoclass:: kani.engines.openai.OpenAIEngine
    :noindex:

Recipes
-------

OpenAI-Compatible Server
^^^^^^^^^^^^^^^^^^^^^^^^
To use an OpenAI-compatible server hosting a non-OpenAI model, you will need to pass the ``api_base`` to the
``OpenAIEngine`` constructor and a ``tokenizer`` for the model.

The ``tokenizer`` should be any object with a method of signature ``.encode(text: str) -> list[Any]``.

.. code-block:: python

    from kani.engines.openai import OpenAIEngine
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("...")
    engine = OpenAIEngine(
        model="my-local-model",
        api_key="dummy",
        api_base="http://127.0.0.1:9001/v1",
        tokenizer=tokenizer,
    )
