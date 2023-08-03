+----------------------------------------+------------------------------------+------------------------------------+-----------------------+
| Model Name                             | Extra                              | Engine                             | Capabilities          |
+========================================+====================================+====================================+=======================+
| GPT-3.5-turbo, GPT-4                   | ``openai``                         | :class:`.OpenAIEngine`             | |function| |api|      |
+----------------------------------------+------------------------------------+------------------------------------+-----------------------+
| |:hugging:| transformers\ [#abstract]_ | ``huggingface``\ [#torch]_         | :class:`.HuggingEngine`            | (abstract)            |
+----------------------------------------+------------------------------------+------------------------------------+-----------------------+
| |:hugging:| |:llama:| LLaMA v2         | ``huggingface, llama``\ [#torch]_  | :class:`.LlamaEngine`              | |oss| |cpu| |gpu|     |
+----------------------------------------+------------------------------------+------------------------------------+-----------------------+
| |:hugging:| |:llama:| Vicuna v1.3      | ``huggingface, llama``\ [#torch]_  | :class:`.VicunaEngine`             | |oss| |cpu| |gpu|     |
+----------------------------------------+------------------------------------+------------------------------------+-----------------------+
| ctransformers\ [#abstract]_            | ``ctransformers``                  | :class:`.CTransformersEngine`      | (abstract)            |
+----------------------------------------+------------------------------------+------------------------------------+-----------------------+
| |:llama:| LLaMA v2 (GGML)              | ``ctransformers``                  | :class:`.LlamaCTransformersEngine` | |oss| |cpu| |gpu|     |
+----------------------------------------+------------------------------------+------------------------------------+-----------------------+

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
