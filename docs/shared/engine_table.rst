+----------------------------------------+----------------------------+-------------------------+
| Model Name                             | Extra                      | Engine                  |
+========================================+============================+=========================+
| gpt-3.5-turbo, gpt-4                   | ``openai``                 | :class:`.OpenAIEngine`  |
+----------------------------------------+----------------------------+-------------------------+
| |:hugging:| transformers\ [#abstract]_ | ``huggingface``\ [#torch]_ | :class:`.HuggingEngine` |
+----------------------------------------+----------------------------+-------------------------+
| |:hugging:| lmsys/vicuna-7b-v1.3       | ``huggingface``\ [#torch]_ | :class:`.VicunaEngine`  |
+----------------------------------------+----------------------------+-------------------------+

.. [#torch] You will also need to install `PyTorch <https://pytorch.org/get-started/locally/>`_.
.. [#abstract] This is an abstract class of models; kani includes a couple concrete |:hugging:| implementations for
  reference.
