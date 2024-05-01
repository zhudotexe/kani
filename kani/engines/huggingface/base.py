import functools
import logging
import warnings
from threading import Thread
from typing import AsyncIterable

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage
from kani.prompts.pipeline import PromptPipeline
from ..base import BaseCompletion, BaseEngine, Completion

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
except ImportError:
    raise MissingModelDependencies(
        'The HuggingEngine requires extra dependencies. Please install kani with "pip install kani[huggingface]". '
        "You will also need to install PyTorch manually."
    ) from None

log = logging.getLogger(__name__)


class HuggingEngine(BaseEngine):
    """Base engine for all HuggingFace text-generation models.

    This class implements the main decoding logic for any HuggingFace model based on a pretrained
    ``AutoModelForCausalLM``. As most models use model-specific chat templates, this base class accepts a
    :class:`.PromptPipeline` to translate kani ChatMessages into a model-specific string.

    **GPU Support**

    By default, the HuggingEngine loads the model on GPU if CUDA is detected on your system. To override the device
    the model is loaded on, pass ``device="cpu|cuda"`` to the constructor.

    .. tip:: See :ref:`4b_quant` for information about loading a quantized model for lower memory usage.
    """

    def __init__(
        self,
        model_id: str,
        max_context_size: int = None,
        prompt_pipeline: PromptPipeline[str | torch.Tensor] = None,
        *,
        token=None,
        device: str | None = None,
        tokenizer_kwargs: dict = None,
        model_load_kwargs: dict = None,
        **hyperparams,
    ):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model. If not given, will be set from the model's config.
        :param prompt_pipeline: The pipeline to translate a list of kani ChatMessages into the model-specific chat
            format (see :class:`.PromptPipeline`).
        :param token: The Hugging Face access token (for gated models). Pass True to load from huggingface-cli.
        :param device: The hardware device to use. If not specified, uses CUDA if available; otherwise uses CPU.
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoTokenizer.from_pretrained()``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if model_load_kwargs is None:
            model_load_kwargs = {}

        tokenizer_kwargs.setdefault("token", hyperparams.get("use_auth_token", token))
        model_load_kwargs.setdefault("token", hyperparams.pop("use_auth_token", token))

        self.model_id = model_id
        self.max_context_size = max_context_size
        self.pipeline = prompt_pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
        self.hyperparams = hyperparams

        # ensure model is on correct device
        if device is None:
            device = "cuda" if torch.backends.cuda.is_built() else "cpu"
        self.device = device
        if self.model.device.type != self.device:
            self.model.to(device)

        # token counting stuff
        # try and infer max context size from the model config if not specified
        if self.max_context_size is None:
            self.max_context_size = getattr(
                self.model.config,
                "model_max_len",
                getattr(self.model.config, "max_position_embeddings", None),
            )
            log.debug(f"Inferred max context size: {self.max_context_size}")

            if self.max_context_size is None:
                raise ValueError(
                    "Could not infer the model's max context size from the config. Please pass the `max_context_size`"
                    " arg."
                )
            elif self.max_context_size > 1e20:
                warnings.warn(
                    f"The inferred max context size of this model is extremely large ({self.max_context_size}). This"
                    " may mean that the model has not configured their model_max_len correctly (or you are still using"
                    " my code in 2050). Please pass the `max_context_size` arg to use the correct model size."
                )

        # infer the token reserve from the pipeline
        if self.token_reserve == 0 and self.pipeline:
            self.token_reserve = self._infer_token_reserve()

    def _infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        prompt = self.pipeline.execute([], for_measurement=True)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokenized)

    def message_len(self, message: ChatMessage) -> int:
        # default concrete base behaviour:
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the HuggingEngine to use it as a non-abstract class."
            )
        prompt = self.pipeline.execute([message], for_measurement=True)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokenized)

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        # default concrete base behaviour:
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the HuggingEngine to use it as a non-abstract class."
            )
        prompt = self.pipeline.execute([], functions, for_measurement=True)
        if isinstance(prompt, torch.Tensor):
            toklen = len(prompt[0])
        else:
            # prompt str to tokens
            tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
            toklen = len(tokenized)

        # warn if there are functions but no tokens
        if functions and toklen == 0:
            warnings.warn(
                "Functions were given to the model, but the function prompt returned 0 tokens! This model may not"
                " support function calling, or you may need to implement"
                f" `{type(self).__name__}.function_token_reserve()`."
            )

        return toklen

    def build_prompt(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None
    ) -> str | torch.Tensor:
        """
        Given the list of messages from kani, build either a single string representing the prompt for the model,
        or build the token tensor.

        The default behaviour is to call the supplied pipeline.
        """
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the HuggingEngine to use it as a non-abstract class."
            )
        prompt = self.pipeline(messages, functions)
        log.debug(f"BUILT PROMPT: {prompt}")
        return prompt

    def _get_generate_args(self, prompt: str | torch.Tensor, **hyperparams):
        """Internal method to build common params for the generate call"""
        if isinstance(prompt, str):
            # prompt str to tokens
            tokenized = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            input_toks = tokenized
            input_len = len(tokenized[0])
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
        return input_toks, input_len, hyperparams

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
        input_toks, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)

        # run it through the model
        output = self.model.generate(input_toks, **hyperparams)
        # decode to tokens
        # the completion shouldn't include the prompt or stop token
        content = self.tokenizer.decode(output[0][input_len:-1]).strip()
        return Completion(
            ChatMessage.assistant(content), prompt_tokens=input_len, completion_tokens=len(output[0]) - (input_len + 1)
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        *,
        streamer_timeout=None,
        **hyperparams,
    ) -> AsyncIterable[str | BaseCompletion]:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param streamer_timeout: The maximum number of seconds to wait for the next token when streaming.
        :param hyperparams: Any additional parameters to pass to GenerationMixin.generate(). (See
            https://huggingface.co/docs/transformers/main_classes/text_generation)
        """
        prompt = self.build_prompt(messages, functions)
        input_toks, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=streamer_timeout
        )

        # run it through the model in another thread so that we can get the tokens in this thread
        generate_func = functools.partial(self.model.generate, input_toks, streamer=streamer, **hyperparams)
        thread = Thread(target=generate_func)
        thread.start()

        # then wait for tokens from the task
        for token in streamer:
            yield token

        # finally clean up the thread
        thread.join()
