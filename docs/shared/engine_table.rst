+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| Model Name                             | Extra                              | Capabilities                 | Engine                                                               |
+========================================+====================================+==============================+======================================================================+
| GPT-3.5-turbo, GPT-4                   | ``openai``                         | |function| |api|             | :class:`kani.engines.openai.OpenAIEngine`                            |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| Claude, Claude Instant                 | ``anthropic``                      | |function| |api|             | :class:`kani.engines.anthropic.AnthropicEngine`                      |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| |:hugging:| transformers\ [#runtime]_  | ``huggingface``\ [#torch]_         | (runtime)                    | :class:`kani.engines.huggingface.HuggingEngine`                      |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| |:hugging:| Command R, Command R+      | ``huggingface``\ [#torch]_         | |function| |oss| |cpu| |gpu| | :class:`kani.engines.huggingface.cohere.CommandREngine`              |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| |:hugging:| |:llama:| LLaMA v2         | ``huggingface, llama``\ [#torch]_  | |oss| |cpu| |gpu|            | :class:`kani.engines.huggingface.llama2.LlamaEngine`                 |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| |:hugging:| |:llama:| Vicuna v1.3      | ``huggingface, llama``\ [#torch]_  | |oss| |cpu| |gpu|            | :class:`kani.engines.huggingface.vicuna.VicunaEngine`                |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| llama.cpp\ [#runtime]_                 | ``cpp``                            | (runtime)                    | :class:`kani.engines.llamacpp.LlamaCppEngine`                        |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+
| |:llama:| LLaMA v2 (GGUF)              | ``cpp``                            | |oss| |cpu| |gpu|            | :class:`kani.engines.llamacpp.LlamaCppEngine`                        |
+----------------------------------------+------------------------------------+------------------------------+----------------------------------------------------------------------+

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

.. [#torch] You will also need to install `PyTorch <https://pytorch.org/get-started/locally/>`_ manually.
.. [#abstract] This is an abstract class of models; kani includes a couple concrete implementations for
  reference.
.. [#runtime] This is a model runtime that can support multiple models using a :class:`.PromptPipeline`.