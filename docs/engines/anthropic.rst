AnthropicEngine
===============
The :class:`.AnthropicEngine` is used to make requests to the Anthropic API.

**TL;DR**

.. code-block:: python

    # see https://docs.anthropic.com/claude/docs/models-overview for a list of model IDs
    from kani.engines.anthropic import AnthropicEngine
    engine = AnthropicEngine(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-sonnet-4-0")

Reference
---------

.. autoclass:: kani.engines.anthropic.AnthropicEngine
    :noindex:

.. autoclass:: kani.engines.anthropic.AnthropicUnknownPart
    :noindex:
    :members:
    :exclude-members: model_config, model_fields, model_computed_fields
    :class-doc-from: class

Recipes
-------

PDF File Inputs
^^^^^^^^^^^^^^^

.. code-block:: python

    from kani import Kani
    from kani.engines.anthropic import AnthropicEngine
    from kani.ext.multimodal_core import BinaryFilePart

    engine = AnthropicEngine(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-sonnet-4-0")
    ai = Kani(engine)

    msg = await ai.chat_round([
        BinaryFilePart.from_file("/path/to/file.pdf"),
        "Summarize this PDF."
    ])
    print(msg.text)
