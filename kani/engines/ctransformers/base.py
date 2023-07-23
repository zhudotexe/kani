import abc

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage
from ..base import BaseEngine, Completion

try:
    from ctransformers import AutoModelForCausalLM
except ImportError as e:
    raise MissingModelDependencies(
        'The CTransformersEngine requires extra dependencies. Please install kani with "pip install kani[ctransformers]".'
    ) from None

class CTransformersEngine(BaseEngine, abc.ABC):
    """CTransformers Engine

    This class implements the main decoding logic for any GGML model based on a pretrained ``AutoModelForCausalLM``.
    """

    def __init__(
        self,
        model_id: str,
        model_file: str = None,
        max_context_size: int = 1024,
        model_load_kwargs: dict = None,
        **hyperparams,
    ):
        """
        :param model_id: The ID of the model to load from HuggingFace or locally.
        :param model_file: The file of the model to load from HuggingFace repo or locally.
        :param max_context_size: The context size of the model.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param hyperparams: Additional arguments to supply the model during generation.
        """

        if model_load_kwargs is None:
            model_load_kwargs = {}

        self.model_id = model_id
        self.model_file = model_file
        self.max_context_size = max_context_size
        model_load_kwargs.setdefault("context_length", self.max_context_size)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, model_file=model_file, **model_load_kwargs)
        self.hyperparams = hyperparams

    @abc.abstractmethod
    def build_prompt(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None
    ) -> str:
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
            https://github.com/marella/ctransformers#method-llmgenerate)
        """
        prompt = self.build_prompt(messages, functions)
        tokens = prompt
        # tokens = self.model.tokenize(prompt)

        content = ''.join(self.model.detokenize(token) for token in self.model.generate(tokens, **hyperparams))

        input_len = len(tokens)
        output_len = len(content)

        return Completion(ChatMessage.assistant(content), prompt_tokens=input_len, completion_tokens=output_len)
