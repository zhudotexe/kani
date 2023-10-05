from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole
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
    """Implementation of Vicuna (a LLaMA v1 fine-tune) using huggingface transformers.

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

    # all prompts start with a hidden <s> token and ASSISTANT:
    token_reserve = 7
    """The Vicuna prompt starts with a hidden <s> token and "ASSISTANT:", which uses 7 tokens."""

    def __init__(self, model_id: str = "lmsys/vicuna-7b-v1.3", *args, **kwargs):
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        tokenizer_kwargs.setdefault("use_fast", False)

        model_load_kwargs = kwargs.pop("model_load_kwargs", {})
        model_load_kwargs.setdefault("low_cpu_mem_usage", _low_cpu_mem_usage)

        kwargs.setdefault("max_context_size", 2048)  # LLaMA has 2048 token window
        super().__init__(
            model_id, *args, tokenizer_kwargs=tokenizer_kwargs, model_load_kwargs=model_load_kwargs, **kwargs
        )

    def build_prompt(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None) -> str:
        prompt_lines = []
        for message in messages:
            if message.role == ChatRole.USER:
                prompt_lines.append(f"USER: {message.text}")
            elif message.role == ChatRole.ASSISTANT:
                prompt_lines.append(f"ASSISTANT: {message.text}</s>")
            else:
                prompt_lines.append(f"{message.text}\n")
        prompt = "\n".join(prompt_lines)
        return f"{prompt}\nASSISTANT:"

    def message_len(self, message: ChatMessage) -> int:
        # https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#example-prompt-weights-v11-and-v13
        # remove 1 for the <s> token at the start
        if message.role == ChatRole.USER:
            # USER: {}\n -> 5
            return self.tokenizer(message.text, return_length=True).length + 4
        elif message.role == ChatRole.ASSISTANT:
            # ASSISTANT: {}</s>\n -> 8
            return self.tokenizer(message.text, return_length=True).length + 7
        # {}\n\n -> 2
        return self.tokenizer(message.text, return_length=True).length + 1
