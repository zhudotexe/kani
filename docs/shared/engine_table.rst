+----------------------------------------+------------------------------------+-----------------------+----------------------------------------------------------------------+
| Model Name                             | Extra                              | Capabilities          | Engine                                                               |
+========================================+====================================+=======================+======================================================================+
| GPT-3.5-turbo, GPT-4                   | ``openai``                         | |function| |api|      | :class:`kani.engines.openai.OpenAIEngine`                            |
+----------------------------------------+------------------------------------+-----------------------+----------------------------------------------------------------------+
| Claude, Claude Instant                 | ``anthropic``                      | |api|                 | :class:`kani.engines.anthropic.AnthropicEngine`                      |
+----------------------------------------+------------------------------------+-----------------------+----------------------------------------------------------------------+
| |:hugging:| transformers\ [#abstract]_ | ``huggingface``\ [#torch]_         | (abstract)            | :class:`kani.engines.huggingface.HuggingEngine`                      |
+----------------------------------------+------------------------------------+-----------------------+----------------------------------------------------------------------+
| |:hugging:| |:llama:| LLaMA v2         | ``huggingface, llama``\ [#torch]_  | |oss| |cpu| |gpu|     | :class:`kani.engines.huggingface.llama2.LlamaEngine`                 |
+----------------------------------------+------------------------------------+-----------------------+----------------------------------------------------------------------+
| |:hugging:| |:llama:| Vicuna v1.3      | ``huggingface, llama``\ [#torch]_  | |oss| |cpu| |gpu|     | :class:`kani.engines.huggingface.vicuna.VicunaEngine`                |
+----------------------------------------+------------------------------------+-----------------------+----------------------------------------------------------------------+
| ctransformers\ [#abstract]_            | ``ctransformers``                  | (abstract)            | :class:`kani.engines.ctransformers.CTransformersEngine`              |
+----------------------------------------+------------------------------------+-----------------------+----------------------------------------------------------------------+
| |:llama:| LLaMA v2 (GGML)              | ``ctransformers``                  | |oss| |cpu| |gpu|     | :class:`kani.engines.ctransformers.llama2.LlamaCTransformersEngine`  |
+----------------------------------------+------------------------------------+-----------------------+----------------------------------------------------------------------+

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
