import abc

from kani.exceptions import MissingModelDependencies
from ..base import BaseEngine

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise MissingModelDependencies(
        'The HuggingEngine requires extra dependencies. Please install kani with "pip install kani[huggingface]". '
        'You will also need to install PyTorch manually.'
    ) from None


class HuggingEngine(BaseEngine, abc.ABC):
    """Base engine for all huggingface text-generation models."""

    def __init__(
        self,
        model_id: str,
        max_context_size: int,
        device: str | None = None,
        tokenizer_kwargs: dict = None,
        model_load_kwargs: dict = None,
        **hyperparams,
    ):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model.
        :param device: The hardware device to use. If not specified, uses CUDA if available; otherwise uses CPU.
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoTokenizer.from_pretrained()``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if model_load_kwargs is None:
            model_load_kwargs = {}
        self.model_id = model_id
        self.max_context_size = max_context_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
        self.hyperparams = hyperparams

        if device is None:
            device = "cuda" if torch.has_cuda else "cpu"
        self.device = device
        self.model.to(device)
