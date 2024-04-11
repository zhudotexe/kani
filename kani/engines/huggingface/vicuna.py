from kani.exceptions import MissingModelDependencies
from kani.prompts.impl import VICUNA_PIPELINE
from .base import HuggingEngine

try:
    import sentencepiece
except ImportError:
    raise MissingModelDependencies(
        'The VicunaEngine requires extra dependencies. Please install kani with "pip install'
        " 'kani[huggingface,llama]'\". You will also need to install PyTorch manually."
    ) from None

try:
    import accelerate

    _low_cpu_mem_usage = True
except ImportError:
    _low_cpu_mem_usage = False


class VicunaEngine(HuggingEngine):
    """
    Implementation of Vicuna (a LLaMA v1 fine-tune) using huggingface transformers.

    You may also use the 13b, 33b, or other LLaMA models that use the Vicuna prompt by passing the HuggingFace model
    ID to the initializer.

    **GPU Support**

    By default, the HuggingEngine loads the model on GPU if CUDA is detected on your system. To override the device
    the model is loaded on, pass ``device="cpu|cuda"`` to the constructor.

    .. seealso:: https://huggingface.co/lmsys/vicuna-7b-v1.3

    .. tip:: See :ref:`4b_quant` for information about loading a quantized model for lower memory usage.

    .. code-block:: python

        engine = VicunaEngine("lmsys/vicuna-7b-v1.3")
        ai = Kani(engine)
    """

    def __init__(self, model_id: str = "lmsys/vicuna-7b-v1.3", *args, **kwargs):
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        tokenizer_kwargs.setdefault("use_fast", False)

        model_load_kwargs = kwargs.pop("model_load_kwargs", {})
        model_load_kwargs.setdefault("low_cpu_mem_usage", _low_cpu_mem_usage)

        kwargs.setdefault("prompt_pipeline", VICUNA_PIPELINE)
        super().__init__(
            model_id, *args, tokenizer_kwargs=tokenizer_kwargs, model_load_kwargs=model_load_kwargs, **kwargs
        )
