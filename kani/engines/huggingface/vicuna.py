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

    @staticmethod
    def build_prompt(messages: list[ChatMessage], functions: list[AIFunction]) -> str:
        if functions:
            warnings.warn("The VicunaEngine is conversational only and does not support function calling.")
        prompt_lines = []
        for message in messages:
            if message.role == ChatRole.USER:
                prompt_lines.append(f"USER: {message.content}")
            elif message.role == ChatRole.ASSISTANT:
                prompt_lines.append(f"ASSISTANT: {message.content}</s>")
            else:
                prompt_lines.append(f"{message.content}\n")
        prompt = "\n".join(prompt_lines)
        return f"{prompt}\nASSISTANT: "

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
        prompt = self.build_prompt(messages, functions)
        # prompt str to tokens
        tokenized = self.tokenizer(prompt, return_tensors="pt", return_length=True)
        input_toks = tokenized.input_ids
        input_toks.to(self.device)
        # set up hyperparams for HF decode
        hyperparams = {**self.hyperparams, **hyperparams}
        hyperparams.setdefault("max_length", self.max_context_size)
        if hyperparams:
            hyperparams.setdefault("do_sample", True)
        # decode to tokens
        output = self.model.generate(inputs=input_toks, **hyperparams)
        content = self.tokenizer.decode(output[0])
        return Completion(
            ChatMessage.assistant(content), prompt_tokens=tokenized.length, completion_tokens=len(output[0])
        )
