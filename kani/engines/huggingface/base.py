import abc

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage
from ..base import BaseEngine, Completion

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise MissingModelDependencies(
        'The HuggingEngine requires extra dependencies. Please install kani with "pip install kani[huggingface]". '
        "You will also need to install PyTorch manually."
    ) from None


class HuggingEngine(BaseEngine, abc.ABC):
    """Base engine for all HuggingFace text-generation models.

    This class implements the main decoding logic for any HuggingFace model based on a pretrained
    ``AutoModelForCausalLM``. To implement a new HuggingFace model, just implement :meth:`~.HuggingEngine.build_prompt`
    and :meth:`~.BaseEngine.message_len` for the specified model.

    **GPU Support**

    By default, the HuggingEngine loads the model on GPU if CUDA is detected on your system. To override the device
    the model is loaded on, pass ``device="cpu|cuda"`` to the constructor.

    .. tip:: See :ref:`4b_quant` for information about loading a quantized model for lower memory usage.
    """

    def __init__(
        self,
        model_id: str,
        max_context_size: int,
        use_auth_token=None,
        device: str | None = None,
        tokenizer_kwargs: dict = None,
        model_load_kwargs: dict = None,
        **hyperparams,
    ):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model.
        :param use_auth_token: The Hugging Face access token (for gated models). Pass True to load from huggingface-cli.
        :param device: The hardware device to use. If not specified, uses CUDA if available; otherwise uses CPU.
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoTokenizer.from_pretrained()``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if model_load_kwargs is None:
            model_load_kwargs = {}

        tokenizer_kwargs.setdefault("use_auth_token", use_auth_token)
        model_load_kwargs.setdefault("use_auth_token", use_auth_token)

        self.model_id = model_id
        self.max_context_size = max_context_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
        self.hyperparams = hyperparams

        if device is None:
            device = "cuda" if torch.has_cuda else "cpu"
        self.device = device
        if self.model.device.type != self.device:
            self.model.to(device)

    @abc.abstractmethod
    def build_prompt(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None
    ) -> str | torch.Tensor:
        """Given the list of messages from kani, build either a single string representing the prompt for the model,
        or build the token tensor."""
        raise NotImplementedError

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param hyperparams: Any additional parameters to pass to GenerationMixin.generate(). (See
            https://huggingface.co/docs/transformers/main_classes/text_generation)
        """
        prompt = self.build_prompt(messages, functions)
        if isinstance(prompt, str):
            # prompt str to tokens
            tokenized = self.tokenizer(prompt, return_tensors="pt", return_length=True)
            input_len = int(tokenized.length)
            input_toks = tokenized.input_ids
        elif isinstance(prompt, torch.Tensor):
            input_toks = prompt
            input_len = len(input_toks[0])
        else:
            raise TypeError("build_prompt should either return a str or a Tensor.")
        # move the input tensor to the right device
        if input_toks.device.type != self.device:
            input_toks = input_toks.to(self.device)
        # set up hyperparams for HF decode
        hyperparams = {**self.hyperparams, **hyperparams}
        hyperparams.setdefault("max_length", self.max_context_size)  # by default HF sets this to 20, which is too small
        # run it through the model
        output = self.model.generate(input_toks, **hyperparams)
        # decode to tokens
        # the completion shouldn't include the prompt or stop token
        content = self.tokenizer.decode(output[0][input_len:-1]).strip()
        return Completion(
            ChatMessage.assistant(content), prompt_tokens=input_len, completion_tokens=len(output[0]) - (input_len + 1)
        )
