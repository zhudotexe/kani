from kani.exceptions import MissingModelDependencies
from kani.prompts.impl import LLAMA2_PIPELINE
from .base import HuggingEngine

try:
    import sentencepiece
    import torch
    from torch import tensor
except ImportError:
    raise MissingModelDependencies(
        'The LlamaEngine requires extra dependencies. Please install kani with "pip install'
        " 'kani[huggingface,llama]'\". You will also need to install PyTorch manually."
    ) from None


class LlamaEngine(HuggingEngine):
    r"""
    Implementation of LLaMA v2 using huggingface transformers.

    You may also use the 13b, 70b, or other LLaMA models that use the LLaMA prompt by passing the HuggingFace model
    ID to the initializer.

    Model IDs:

    - ``meta-llama/Llama-2-7b-chat-hf``
    - ``meta-llama/Llama-2-13b-chat-hf``
    - ``meta-llama/Llama-2-70b-chat-hf``

    In theory you could also use the non-chat-tuned variants as well.

    **GPU Support**

    By default, the HuggingEngine loads the model on GPU if CUDA is detected on your system. To override the device
    the model is loaded on, pass ``device="cpu|cuda"`` to the constructor.

    **Usage**

    .. code-block:: python

        engine = LlamaEngine("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)
        ai = Kani(engine)

    .. attention::

        You will need to accept Meta's license in order to download the LLaMA v2 weights. Visit
        https://ai.meta.com/resources/models-and-libraries/llama-downloads/ and
        https://huggingface.co/meta-llama/Llama-2-7b-chat-hf to request access.

        Then, run ``huggingface-cli login`` to authenticate with Hugging Face.

    .. seealso:: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

    .. tip:: See :ref:`4b_quant` for information about loading a quantized model for lower memory usage.

    .. tip::

        This engine is equivalent to the following usage of the base :class:`.HuggingEngine`.

        .. code-block:: python

            LLAMA2_PIPELINE = (
                PromptPipeline()
                .wrap(role=ChatRole.SYSTEM, prefix="<<SYS>>\n", suffix="\n<</SYS>>\n")
                .translate_role(role=ChatRole.SYSTEM, to=ChatRole.USER)
                .merge_consecutive(role=ChatRole.USER, sep="\n")
                .merge_consecutive(role=ChatRole.ASSISTANT, sep=" ")
                .conversation_fmt(
                    user_prefix="<s>[INST] ",
                    user_suffix=" [/INST]",
                    assistant_prefix=" ",
                    assistant_suffix=" </s>",
                    assistant_suffix_if_last="",
                )
            )

            engine = HuggingEngine(
                "meta-llama/Llama-2-7b-chat-hf",
                prompt_pipeline=LLAMA2_PIPELINE
            )

        See :class:`.PromptPipeline` for more information on reusable prompt pipelines.
    """

    def __init__(self, model_id: str = "meta-llama/Llama-2-7b-chat-hf", *args, **kwargs):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param token: The Hugging Face access token (for gated models). Pass True to load from huggingface-cli.
        :param max_context_size: The context size of the model.
        :param device: The hardware device to use. If not specified, uses CUDA if available; otherwise uses CPU.
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoTokenizer.from_pretrained()``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        kwargs.setdefault("prompt_pipeline", LLAMA2_PIPELINE)
        super().__init__(model_id, *args, **kwargs)
