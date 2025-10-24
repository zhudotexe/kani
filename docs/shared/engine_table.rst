+----------------------------------------+------------------------------------+-------------------------+----------------------------------------------------------------------+
| Model Name                             | Extra                              | Capabilities            | Engine                                                               |
+========================================+====================================+=========================+======================================================================+
| All OpenAI Models                      | ``openai``                         | |function| |multimodal| | :class:`kani.engines.openai.OpenAIEngine`                            |
+----------------------------------------+------------------------------------+-------------------------+----------------------------------------------------------------------+
| All Anthropic Models                   | ``anthropic``                      | |function| |multimodal| | :class:`kani.engines.anthropic.AnthropicEngine`                      |
+----------------------------------------+------------------------------------+-------------------------+----------------------------------------------------------------------+
| All Google AI Models                   | ``google``                         | |function| |multimodal| | :class:`kani.engines.google.GoogleAIEngine`                          |
+----------------------------------------+------------------------------------+-------------------------+----------------------------------------------------------------------+
| |:hugging:| transformers\ [#hf]_       | ``huggingface``\ [#torch]_         | (model-specific)        | :class:`kani.engines.huggingface.HuggingEngine`                      |
+----------------------------------------+------------------------------------+-------------------------+----------------------------------------------------------------------+
| llama.cpp\ [#runtime]_                 | ``cpp``                            | (model-specific)        | :class:`kani.engines.llamacpp.LlamaCppEngine`                        |
+----------------------------------------+------------------------------------+-------------------------+----------------------------------------------------------------------+

Additional models using the classes above are also supported - see the
`model zoo <https://github.com/zhudotexe/kani/blob/main/examples/4_engines_zoo.py>`_ for a more comprehensive list of
models!

**Legend**

- |function|: Supports function calling.
- |multimodal|: Supports multimodal inputs.

.. |function| replace:: :abbr:`🛠️ (supports function calling)`
.. |multimodal| replace:: :abbr:`🖼 (supports multimodal inputs)`

.. [#torch] You will also need to install `PyTorch <https://pytorch.org/get-started/locally/>`_ manually.
.. [#runtime] This is a model runtime that can support multiple models using a :class:`.PromptPipeline`.
.. [#hf] The HuggingEngine can run most models directly from HuggingFace using Chat Templates. For more fine-grained
   control over prompting, see :class:`.PromptPipeline`.
