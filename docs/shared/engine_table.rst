+----------------------------------------+------------------------------------+-------------------------+
| Model Name                             | Extra                              | Engine                  |
+========================================+====================================+=========================+
| GPT-3.5-turbo, GPT-4                   | ``openai``                         | :class:`.OpenAIEngine`  |
+----------------------------------------+------------------------------------+-------------------------+
| |:hugging:| transformers\ [#abstract]_ | ``huggingface``\ [#torch]_         | :class:`.HuggingEngine` |
+----------------------------------------+------------------------------------+-------------------------+
| |:hugging:| |:llama:| LLaMA v2         | ``huggingface, llama``\ [#torch]_  | :class:`.LlamaEngine`   |
+----------------------------------------+------------------------------------+-------------------------+
| |:hugging:| |:llama:| Vicuna v1.3      | ``huggingface, llama``\ [#torch]_  | :class:`.VicunaEngine`  |
+----------------------------------------+------------------------------------+-------------------------+

.. [#torch] You will also need to install `PyTorch <https://pytorch.org/get-started/locally/>`_ manually.
.. [#abstract] This is an abstract class of models; kani includes a couple concrete |:hugging:| implementations for
  reference.
