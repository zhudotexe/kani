import warnings

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole
from .base import HuggingEngine

try:
    import torch
    import sentencepiece
except ImportError:
    raise MissingModelDependencies(
        'The LlamaEngine requires extra dependencies. Please install kani with "pip install'
        ' \'kani[huggingface,llama]\'\". You will also need to install PyTorch manually.'
    ) from None

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class LlamaEngine(HuggingEngine):
    """Implementation of LLaMA v2 using huggingface transformers.

    You may also use the 13b, 70b, or other LLaMA models that use the LLaMA prompt by passing the HuggingFace model
    ID to the initializer.

    Model IDs:

    - ``meta-llama/Llama-2-7b-chat-hf``
    - ``meta-llama/Llama-2-13b-chat-hf``
    - ``meta-llama/Llama-2-70b-chat-hf``

    In theory you could also use the non-chat-tuned variants as well.

    .. attention::

        You will need to accept Meta's license in order to download the LLaMA v2 weights. Visit
        https://ai.meta.com/resources/models-and-libraries/llama-downloads/ and
        https://huggingface.co/meta-llama/Llama-2-7b-chat-hf to request access.

        Then, run ``huggingface-cli login`` to authenticate with Hugging Face.

    .. seealso:: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

    .. code-block:: python

        engine = LlamaEngine("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)
        ai = Kani(engine)
    """

    def __init__(self, model_id: str = "meta-llama/Llama-2-7b-chat-hf", *args, strict=False, **kwargs):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param use_auth_token: The Hugging Face access token (for gated models). Pass True to load from huggingface-cli.
        :param strict: Whether to enforce the same prompt constraints as Meta's LLaMA (always starting with
            system, alternating user/assistant). This is recommended for using the base LLaMA model.
        :param max_context_size: The context size of the model.
        :param device: The hardware device to use. If not specified, uses CUDA if available; otherwise uses CPU.
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoTokenizer.from_pretrained()``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        self.strict = strict

        kwargs.setdefault("max_context_size", 4096)  # LLaMA has 4096 token window
        super().__init__(model_id, *args, **kwargs)

    # noinspection PyMethodMayBeStatic
    def _build_prompt_strict(self, messages: list[ChatMessage]):
        if len(messages) < 2:
            raise ValueError("You must specify at least one message and a system prompt (LLaMA strict mode).")
        if messages[0].role != ChatRole.SYSTEM:
            raise ValueError("The first message must be a system prompt (LLaMA strict mode).")
        if not (
            all([m.role == ChatRole.USER for m in messages[1::2]])
            and all([m.role == ChatRole.ASSISTANT for m in messages[2::2]])
        ):
            raise ValueError("Messages after the first must alternate between user and system (LLaMA strict mode).")
        if messages[-1].role != ChatRole.USER:
            raise ValueError("The last message must be from the user (LLaMA strict mode).")
        # implementation based on llama code
        dialog = [f"{B_SYS}{messages[0].content}{E_SYS}{messages[1].content}"] + [m.content for m in messages[2:]]
        dialog_tokens = sum(
            [
                self.tokenizer.encode(f"{B_INST} {prompt} {E_INST} {answer} </s>")
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        dialog_tokens += self.tokenizer.encode(f"{B_INST} {dialog[-1]} {E_INST}")
        return torch.tensor([dialog_tokens], dtype=torch.long)

    def build_prompt(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None) -> torch.Tensor:
        if functions:
            warnings.warn("The LlamaEngine is conversational only and does not support function calling.")
        if self.strict:
            return self._build_prompt_strict(messages)
        # non-strict has to kind of just do its best
        # ganbatte, kani, ganbatte
        prompt_parts = []
        for message in messages:
            if message.role == ChatRole.USER:
                prompt_parts.append(f"<s> {B_INST} {message.content} {E_INST}")
            elif message.role == ChatRole.ASSISTANT:
                prompt_parts.append(f" {message.content} </s>")
            else:
                prompt_parts.append(f"{B_INST} {B_SYS}{message.content}{E_SYS} {E_INST}")
        return self.tokenizer.encode("".join(prompt_parts), add_special_tokens=False, return_tensors="pt")

    def message_len(self, message: ChatMessage) -> int:
        # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212
        if message.role == ChatRole.USER:
            # <s> [INST] {} [/INST] -> 7
            return self.tokenizer(message.content, return_length=True).length[0] + 7
        elif message.role == ChatRole.ASSISTANT:
            # {} </s> -> 2
            return self.tokenizer(f" {message.content} ", return_length=True).length[0] + 2
        # <s> [INST] <<SYS>>\n{}\n<</SYS>>\n\n [/INST] -> 20
        return self.tokenizer(message.content, return_length=True).length[0] + 20
