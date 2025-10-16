import logging
import warnings
from collections import UserDict
from threading import Thread
from typing import AsyncIterable

from kani import _optional, model_specific
from kani.ai_function import AIFunction
from kani.engines.base import BaseCompletion, BaseEngine, Completion
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage
from kani.prompts.pipeline import PromptPipeline
from kani.utils.warnings import deprecated

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        BatchEncoding,
        BatchFeature,
        PreTrainedTokenizerBase,
        ProcessorMixin,
        TextIteratorStreamer,
    )
except ImportError:
    raise MissingModelDependencies(
        'The HuggingEngine requires extra dependencies. Please install kani with "pip install kani[huggingface]". '
        "You will also need to install PyTorch manually."
    ) from None

has_cuda = torch.backends.cuda.is_built()
has_mps = torch.backends.mps.is_built()

try:
    import accelerate

    has_accelerate = True
except ImportError:
    if has_cuda:
        warnings.warn(
            "A PyTorch install with CUDA was detected on this system, but `accelerate` is not installed. Run `pip"
            " install accelerate` for automatic GPU mapping of Hugging Face models."
        )
    has_accelerate = False

log = logging.getLogger(__name__)


class HuggingEngine(BaseEngine):
    """Base engine for all HuggingFace text-generation models.

    This class implements the main decoding logic for any HuggingFace model based on a pretrained
    ``AutoModelForCausalLM``. As most models use model-specific chat templates, this base class accepts a
    :class:`.PromptPipeline` to translate kani ChatMessages into a model-specific string.

    .. versionadded:: 1.2.0
        By default, the ``HuggingEngine`` uses models' bundled chat template to build the prompt
        for chat-based models available on Hugging Face. See
        https://huggingface.co/docs/transformers/main/en/chat_templating for more information.

    **GPU Support**

    By default, the HuggingEngine loads the model on GPU if CUDA is detected on your system. To override the device
    the model is loaded on, pass ``device="cpu|cuda"`` to the constructor.

    **Multimodal support**: audio, images, video (depending on model).

    .. tip:: See :ref:`4b_quant` for information about loading a quantized model for lower memory usage.
    """

    def __init__(
        self,
        model_id: str,
        max_context_size: int = None,
        prompt_pipeline: PromptPipeline[str | torch.Tensor] = None,
        *,
        # hf args
        token=None,
        device: str | None = None,
        tokenizer_cls=None,
        tokenizer_kwargs: dict = None,
        model_cls=AutoModelForCausalLM,
        model_load_kwargs: dict = None,
        chat_template_kwargs: dict = None,
        # multimodal args
        mm_audio_sample_rate: int = None,
        mm_video_fps: float = 1,
        # kani args
        token_reserve: int = 0,
        **hyperparams,
    ):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model. If not given, will be set from the model's config.
        :param prompt_pipeline: The pipeline to translate a list of kani ChatMessages into the model-specific chat
            format (see :class:`.PromptPipeline`). If not passed, uses the Hugging Face chat template if available.
        :param token: The Hugging Face access token (for gated models). Pass True to load from huggingface-cli.
        :param device: The hardware device to use. If not specified, uses CUDA or MPS if available; otherwise uses CPU.
        :param tokenizer_cls: Advanced use cases: The HF tokenizer class to use. Defaults to ``AutoProcessor`` (if no
            processing config is available or this raises an error, this will fall back to ``AutoTokenizer``).
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoProcessor.from_pretrained()``.
        :param model_cls: Advanced use cases: The HF model class to use. Defaults to ``AutoModelForCausalLM``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param chat_template_kwargs: The keyword arguments to pass to ``tokenizer.apply_chat_template`` if using a chat
            template prompt pipeline.
        :param mm_audio_sample_rate: The sample rate to remux audio inputs to. Check your model's documentation for the
            expected sample rate. By default, does not change the sample rate of the input file.
        :param mm_video_fps: The number of image frames to sample per second of video input.
        :param hyperparams: Additional arguments to supply the model during generation.
        :param token_reserve: DEPRECATED: The number of tokens to reserve for internal engine mechanisms (e.g. if there
            is a generation template after the last user message). If not passed, kani will attempt to infer this from a
            prompt pipeline.
        """
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if model_load_kwargs is None:
            model_load_kwargs = {}
        if chat_template_kwargs is None:
            chat_template_kwargs = {}

        tokenizer_kwargs.setdefault("token", hyperparams.get("use_auth_token", token))
        model_load_kwargs.setdefault("token", hyperparams.pop("use_auth_token", token))
        model_load_kwargs.setdefault("torch_dtype", "auto")
        if has_cuda and has_accelerate:
            model_load_kwargs.setdefault("device_map", "auto")

        self.model_id = model_id
        self.max_context_size = max_context_size

        # load the correct processor or tokenizer, EAFP
        if tokenizer_cls is None:
            try:
                _processor_or_tokenizer = AutoProcessor.from_pretrained(model_id, **tokenizer_kwargs)
            except Exception as e:
                log.warning(
                    f"Could not load the AutoProcessor for {model_id}, falling back to AutoTokenizer. Multimodal"
                    " inputs will not be available.",
                    exc_info=e,
                )
                _processor_or_tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        else:
            _processor_or_tokenizer = tokenizer_cls.from_pretrained(model_id, **tokenizer_kwargs)
        self._processor_or_tokenizer: ProcessorMixin | PreTrainedTokenizerBase = _processor_or_tokenizer
        self.model = model_cls.from_pretrained(model_id, **model_load_kwargs)
        self.hyperparams = hyperparams

        # multimodal args
        self.mm_audio_sample_rate = mm_audio_sample_rate
        self.mm_video_fps = mm_video_fps

        # load the pipeline
        if prompt_pipeline is None:
            # try and load a manual impl, or default to chat template if not available
            prompt_pipeline = model_specific.prompt_pipeline_for_hf_model(
                model_id, self._processor_or_tokenizer, chat_template_kwargs=chat_template_kwargs
            )
        self.pipeline = prompt_pipeline

        # ensure model is on correct device
        if device is None:
            if has_cuda:
                device = "cuda"
            elif has_mps:
                device = "mps"
            else:
                device = "cpu"
            log.info(f"Inferred device for model weights: {device}. Set `device=...` if this is incorrect.")
        self.device = device
        if self.model.device.type != self.device:
            self.model.to(device)

        # ensure model is in eval mode
        self.model.eval()

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
                    " may mean that the model has not configured their model_max_len correctly. Please pass the"
                    " `max_context_size` arg to use the correct model size."
                )

        # deprecated
        self._token_reserve = token_reserve

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        # little util to make sure we get a Tokenizer object when we need it
        # just in case we load a processor (for multimodal models)
        if isinstance(self._processor_or_tokenizer, ProcessorMixin):
            # noinspection PyUnresolvedReferences
            return self._processor_or_tokenizer.tokenizer
        return self._processor_or_tokenizer

    def _collect_multimodal(self, messages: list[ChatMessage]):
        """Collect a list of audios, images, videos from the given input prompt."""
        if not _optional.has_multimodal_core:
            return None, None, None
        audios = []
        images = []
        videos = []

        for msg in messages:
            for part in msg.parts:
                if isinstance(part, _optional.multimodal_core.AudioPart):
                    audios.append(part.as_ndarray(sr=self.mm_audio_sample_rate))
                elif isinstance(part, _optional.multimodal_core.ImagePart):
                    images.append(part.image)
                elif isinstance(part, _optional.multimodal_core.VideoPart):
                    videos.append(part.as_tensor(fps=self.mm_video_fps))

        return audios or None, images or None, videos or None

    def build_prompt(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None
    ) -> str | torch.Tensor | BatchEncoding | BatchFeature:
        """
        Given the list of messages from kani, build either a single string representing the prompt for the model,
        or build the token tensor.

        The default behaviour is to call the supplied pipeline.
        """
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the HuggingEngine to use it as a non-abstract class."
            )
        text = self.pipeline(messages, functions)
        log.debug(f"BUILT PROMPT TEXT: {text}")

        # if multimodal is installed and we have a processor, collect parts and run them through the processor
        if _optional.has_multimodal_core and isinstance(self._processor_or_tokenizer, ProcessorMixin):
            audios, images, videos = self._collect_multimodal(messages)
            inputs = self._processor_or_tokenizer(  # should be a processor in this case
                text=text, audio=audios, images=images, videos=videos, add_special_tokens=False, return_tensors="pt"
            )
            return inputs

        # otherwise run it through the processor/tokenizer with just text
        inputs = self._processor_or_tokenizer(text=text, add_special_tokens=False, return_tensors="pt")
        return inputs

    def _get_generate_args(self, prompt: str | torch.Tensor | BatchEncoding | BatchFeature, **hyperparams):
        """
        Internal method to build common params for the generate call
        and also do some chores before we generate
        """
        # make sure the prompt is tokenized
        if isinstance(prompt, str):
            # prompt str to tokens
            input_kwargs = self._processor_or_tokenizer(text=prompt, add_special_tokens=False, return_tensors="pt")
            input_len = input_kwargs["input_ids"].shape[1]
        elif isinstance(prompt, torch.Tensor):
            input_kwargs = BatchFeature({"input_ids": prompt})
            input_len = len(prompt[0])
        elif isinstance(prompt, (dict, UserDict)):
            input_kwargs = prompt
            input_len = input_kwargs["input_ids"].shape[1]
        else:
            raise TypeError(
                "build_prompt should either return a str, Tensor, or dict (e.g., BatchEncoding, BatchFeature)."
            )

        # move the input tensor to the right device and make sure any multimodal features are in the right dtype
        # (if BatchFeature)
        input_kwargs.to(self.device)
        if isinstance(input_kwargs, BatchFeature):
            input_kwargs.to(self.model.dtype)

        # set up hyperparams for HF decode
        hyperparams = {**self.hyperparams, **hyperparams}
        if "max_new_tokens" not in hyperparams:
            hyperparams.setdefault("max_length", self.max_context_size)

        # check for a model-specific parser
        model_specific.warn_for_uninitialized_parser(self.model_id)

        return input_kwargs, input_len, hyperparams

    def _get_eos_tokens(self, *, return_ids=False, **hyperparams) -> list[str] | list[int]:
        """Get the list of tokens that should end a generation."""
        if "eos_token_id" in hyperparams:
            genconfig_eos_token_id = hyperparams["eos_token_id"]
        else:
            genconfig_eos_token_id = self.model.generation_config.eos_token_id

        if isinstance(genconfig_eos_token_id, list):
            eos_token_ids = genconfig_eos_token_id
        elif genconfig_eos_token_id is not None:
            eos_token_ids = [genconfig_eos_token_id]
        else:
            warnings.warn(
                f"No EOS token was found for the {self.model_id} model. Generation may continue forever. Please pass"
                " `eos_token_id=[...]` in the engine constructor."
            )
            eos_token_ids = []
        if return_ids:
            return eos_token_ids
        return [self._processor_or_tokenizer.decode(t) for t in eos_token_ids]

    # ==== kani impl ====
    async def prompt_len(self, messages, functions=None, **kwargs) -> int:
        prompt = self.build_prompt(messages, functions)
        input_kwargs, input_len, _ = self._get_generate_args(prompt, **kwargs)
        return input_len

    async def predict(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        *,
        decode_kwargs: dict = None,
        **hyperparams,
    ) -> Completion:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``prompt_len(messages, functions)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param decode_kwargs: Any arguments to pass to AutoTokenizer.decode().
        :param hyperparams: Any additional parameters to pass to GenerationMixin.generate(). (See
            https://huggingface.co/docs/transformers/main_classes/text_generation)
        """
        if decode_kwargs is None:
            decode_kwargs = {}

        prompt = self.build_prompt(messages, functions)
        input_kwargs, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)
        eos_tok_ids = self._get_eos_tokens(return_ids=True, **hyperparams)

        # run it through the model
        with torch.no_grad():
            output = self.model.generate(**input_kwargs, **hyperparams)
        # decode to tokens
        # the completion shouldn't include the prompt or stop token
        if output[0][-1] in eos_tok_ids:
            content = self._processor_or_tokenizer.decode(output[0][input_len:-1], **decode_kwargs).strip()
            output_len = len(output[0]) - (input_len + 1)
        else:
            content = self._processor_or_tokenizer.decode(output[0][input_len:], **decode_kwargs).strip()
            output_len = len(output[0]) - input_len
        return Completion(ChatMessage.assistant(content), prompt_tokens=input_len, completion_tokens=output_len)

    async def stream(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        *,
        streamer_timeout: float | None = None,
        decode_kwargs: dict = None,
        **hyperparams,
    ) -> AsyncIterable[str | BaseCompletion]:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``prompt_len(messages, functions)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param streamer_timeout: The maximum number of seconds to wait for the next token when streaming.
        :param decode_kwargs: Any arguments to pass to AutoTokenizer.decode().
        :param hyperparams: Any additional parameters to pass to GenerationMixin.generate(). (See
            https://huggingface.co/docs/transformers/main_classes/text_generation)
        """
        if decode_kwargs is None:
            decode_kwargs = {}

        prompt = self.build_prompt(messages, functions)
        input_kwargs, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)
        eos_toks = self._get_eos_tokens(**hyperparams)
        streamer = TextIteratorStreamer(
            self._processor_or_tokenizer, skip_prompt=True, timeout=streamer_timeout, **decode_kwargs
        )

        # run it through the model in another thread so that we can get the tokens in this thread
        output_toks = None

        def thread_target():
            nonlocal output_toks  # ugly way of sending the results of .generate to the outer scope
            with torch.no_grad():
                output_toks = self.model.generate(**input_kwargs, streamer=streamer, **hyperparams)

        thread = Thread(target=thread_target)
        thread.start()

        # then wait for tokens from the task
        yielded_tokens = []
        for token in streamer:
            for eos_tok in eos_toks:
                if token.endswith(eos_tok):
                    token = token[: -len(eos_tok)]
                    break
            if not token:
                continue
            yield token
            yielded_tokens.append(token)

        # clean up the thread
        thread.join()

        # yield a completion with usage stats
        content = "".join(yielded_tokens)
        yield Completion(
            message=ChatMessage.assistant(content=content.strip()),
            prompt_tokens=input_len,
            completion_tokens=len(output_toks[0]) - (input_len + 1),
        )

    # ===== deprecated =====
    @property
    @deprecated("Use prompt_len instead")
    def token_reserve(self):
        # infer the token reserve from the pipeline
        if self._token_reserve == 0 and self.pipeline:
            self._token_reserve = self._infer_token_reserve()
        return self._token_reserve

    def _infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        prompt = self.pipeline.execute([], for_measurement=True)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokenized)

    @deprecated("Use prompt_len instead")
    def message_len(self, message: ChatMessage) -> int:
        """Return the length, in tokens, of the given chat message.

        The HuggingEngine's default implementation renders the message with ``apply_chat_template`` if no
        ``prompt_pipeline`` is supplied.
        """
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

    @deprecated("Use prompt_len instead")
    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0
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
        if toklen == 0:
            warnings.warn(
                "Functions were given to the model, but the function prompt returned 0 tokens! This model may not"
                " support function calling, or you may need to implement"
                f" `{type(self).__name__}.function_token_reserve()`."
            )

        return toklen
