+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| Model Name                             | Extra                              | Capabilities                 | Engine                                                               |
+========================================+====================================+==============================+======================================================================+
| All OpenAI Models                      | ``openai``                         | |function| |api|             | :class:`kani.engines.openai.OpenAIEngine`                            |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| All Anthropic Models                   | ``anthropic``                      | |function| |api|             | :class:`kani.engines.anthropic.AnthropicEngine`                      |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| |:hugging:| transformers\ [#hf]_       | ``huggingface``\ [#torch]_         | (model-specific)             | :class:`kani.engines.huggingface.HuggingEngine`                      |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| |:hugging:| Command R, Command R+      | ``huggingface``\ [#torch]_         | |function| |oss| |cpu| |gpu| | :class:`kani.engines.huggingface.cohere.CommandREngine`              |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| llama.cpp\ [#runtime]_                 | ``cpp``                            | (model-specific)             | :class:`kani.engines.llamacpp.LlamaCppEngine`                        |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+

Additional models using the classes above are also supported - see the
`model zoo <https://github.com/zhudotexe/kani/blob/main/examples/4_engines_zoo.py>`_ for a more comprehensive list of
models!

**Legend**

- |function|: Supports function calling.
- |oss|: Open source model.
- |cpu|: Runs locally on CPU.
- |gpu|: Runs locally on GPU.
- |api|: Hosted API.

.. |function| replace:: :abbr:`🛠️ (supports function calling)`
.. |oss| replace:: :abbr:`🔓(open source)`
.. |cpu| replace:: :abbr:`🖥 (runs on local cpu)`
.. |gpu| replace:: :abbr:`🚀 (runs on local gpu)`
.. |api| replace:: :abbr:`📡 (hosted API)`

.. [#zoo] See the `model zoo <https://github.com/zhudotexe/kani/blob/main/examples/4_engines_zoo.py>`_ for a code sample
   to initialize this model with the given engine.
.. [#torch] You will also need to install `PyTorch <https://pytorch.org/get-started/locally/>`_ manually.
.. [#abstract] This is an abstract class of models; kani includes a couple concrete implementations for
  reference.
.. [#runtime] This is a model runtime that can support multiple models using a :class:`.PromptPipeline`.
.. [#hf] The HuggingEngine can run most models directly from HuggingFace using Chat Templates. For more fine-grained
   control over prompting, see :class:`.PromptPipeline`.
