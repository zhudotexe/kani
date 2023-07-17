import warnings

from kani.ai_function import AIFunction
from kani.models import ChatMessage, ChatRole
from .base import HuggingEngine
from ..base import Completion


class VicunaEngine(HuggingEngine):
    """Implementation of Vicuna using huggingface transformers.

    https://huggingface.co/lmsys/vicuna-7b-v1.3

    .. code-block:: python

        engine = VicunaEngine("lmsys/vicuna-7b-v1.3")
        ai = Kani(engine)
    """

    def __init__(self, model_id: str = "lmsys/vicuna-7b-v1.3", *args, **kwargs):
        if "tokenizer_kwargs" in kwargs:
            kwargs["tokenizer_kwargs"]["use_fast"] = False
        if "model_load_kwargs" in kwargs:
            kwargs["model_load_kwargs"]["low_cpu_mem_usage"] = True
        kwargs.setdefault("max_context_size", 2048)  # LLaMA has 2048 token window
        super().__init__(model_id, *args, **kwargs)

    def message_len(self, message: ChatMessage) -> int:
        # https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#example-prompt-weights-v11-and-v13
        if message.role == ChatRole.USER:
            # USER: {}\n -> 6
            return self.tokenizer(message.content, return_length=True).length + 6
        elif message.role == ChatRole.ASSISTANT:
            # ASSISTANT: {}</s>\n -> 6
            return self.tokenizer(message.content, return_length=True).length + 10
        # {}\n\n -> 2
        return self.tokenizer(message.content, return_length=True).length + 2

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        if functions:
            warnings.warn("The VicunaEngine is conversational only and does not support function calling.")
        tokenized = self.tokenizer(prompt_template, return_tensors="pt", return_length=True)
        input_toks = tokenized.input_ids
        input_toks.to(self.device)
        output = self.model.generate(inputs=input_toks, **self.hyperparams, **hyperparams)
        content = self.tokenizer.decode(output[0])
        return Completion(
            ChatMessage.assistant(content), prompt_tokens=tokenized.length, completion_tokens=len(output[0])
        )
