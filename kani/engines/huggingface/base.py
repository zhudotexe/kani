import logging
import warnings
from collections import defaultdict
from functools import cached_property
from threading import Thread
from typing import AsyncIterable

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage
from kani.prompts.pipeline import PromptPipeline
from ..base import BaseCompletion, BaseEngine, Completion
from ... import ChatRole

try:
    import torch
    import transformers
    from jinja2 import TemplateError
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

    .. versionadded:: 0.8.0
        The ``HuggingEngine`` is no longer abstract - it will now use models' bundled chat template to build the prompt
        for chat-based models available on Hugging Face. See
        https://huggingface.co/docs/transformers/main/en/chat_templating for more information.

        Requires ``transformers>=4.34.0``.

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
        # hf args
        token=None,
        device: str | None = None,
        tokenizer_kwargs: dict = None,
        model_load_kwargs: dict = None,
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
        :param device: The hardware device to use. If not specified, uses CUDA if available; otherwise uses CPU.
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoTokenizer.from_pretrained()``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param hyperparams: Additional arguments to supply the model during generation.
        :param token_reserve: The number of tokens to reserve for internal engine mechanisms (e.g. if there is a
            generation template after the last user message). If not passed, kani will attempt to infer this from a
            prompt pipeline.
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
        self.token_reserve = token_reserve

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
                    " may mean that the model has not configured their model_max_len correctly. Please pass the"
                    " `max_context_size` arg to use the correct model size."
                )

        # infer the token reserve from the pipeline
        if self.token_reserve == 0 and self.pipeline:
            self.token_reserve = self._infer_token_reserve()

        # no pipeline estimation caches
        self._padding_len_by_role: dict[ChatRole, int] = defaultdict(lambda: 0)
        if self.token_reserve == 0 and not self.pipeline:
            self.token_reserve = self._chat_template_infer_token_reserve()

    _chat_template_dummy_msg = {"role": "user", "content": "dummy"}

    @cached_property
    def _chat_template_dummy_len(self) -> int:
        return len(self.tokenizer.apply_chat_template([self._chat_template_dummy_msg], add_generation_prompt=False))

    def _chat_template_infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        full_len = self.tokenizer.apply_chat_template([self._chat_template_dummy_msg], add_generation_prompt=True)
        return full_len - self._chat_template_dummy_len

    def _infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        prompt = self.pipeline.execute([], for_measurement=True)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokenized)

    def _chat_template_message_len(self, message: ChatMessage) -> int:
        """Estimate the message length of a single message based off the chat template."""
        _ensure_chat_template(self.tokenizer)
        conversation = [{"role": message.role.value, "content": message.text}]
        try:
            return len(self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False))
        except TemplateError:
            # the template probably enforces user/assistant,
            # return a best-effort estimate based on the cached additions to messages of this role
            raw_tok_len = len(self.tokenizer.encode(message.text, add_special_tokens=False))
            return raw_tok_len + self._padding_len_by_role[message.role]

    def message_len(self, message: ChatMessage) -> int:
        """Return the length, in tokens, of the given chat message.

        The HuggingEngine's default implementation renders the message with ``apply_chat_template`` if no
        ``prompt_pipeline`` is supplied.
        """
        # default concrete base behaviour:
        if self.pipeline is None:
            return self._chat_template_message_len(message)
            # raise NotImplementedError(
            #     "You must pass a prompt_pipeline to the HuggingEngine to use it as a non-abstract class."
            # )
        prompt = self.pipeline.execute([message], for_measurement=True)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokenized)

    def _chat_template_function_token_reserve(self, functions: list[AIFunction]) -> int:
        """Estimate the function token reserve based off the chat template."""
        _ensure_chat_template(self.tokenizer)
        tools = [f.json_schema for f in functions]
        full_len = len(
            self.tokenizer.apply_chat_template(
                [self._chat_template_dummy_msg], tools=tools, add_generation_prompt=False
            )
        )
        return full_len - self._chat_template_dummy_len

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        # default concrete base behaviour:
        if self.pipeline is None:
            return self._chat_template_function_token_reserve(functions)
            # raise NotImplementedError(
            #     "You must pass a prompt_pipeline to the HuggingEngine to use it as a non-abstract class."
            # )
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

    def _chat_template_build_prompt(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None
    ) -> str | torch.Tensor:
        """Given the list of messages from kani, build either a single string representing the prompt for the model,
        or build the token tensor.

        The default implementation uses the model tokenizer's `apply_chat_template` method.
        """
        _ensure_chat_template(self.tokenizer)
        conversation = [{"role": msg.role.value, "content": msg.text} for msg in messages]
        tools = [f.json_schema for f in functions]
        return self.tokenizer.apply_chat_template(
            conversation, tools=tools, add_generation_prompt=True, return_tensors="pt"
        )

    def build_prompt(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None
    ) -> str | torch.Tensor:
        """
        Given the list of messages from kani, build either a single string representing the prompt for the model,
        or build the token tensor.

        The default behaviour is to call the supplied pipeline.
        """
        if self.pipeline is None:
            prompt = self._chat_template_build_prompt(messages, functions)
            # raise NotImplementedError(
            #     "You must pass a prompt_pipeline to the HuggingEngine to use it as a non-abstract class."
            # )
        else:
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
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        *,
        decode_kwargs: dict = None,
        **hyperparams,
    ) -> Completion:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param decode_kwargs: Any arguments to pass to AutoTokenizer.decode(). Defaults to
            ``dict(skip_special_tokens=True)``.
        :param hyperparams: Any additional parameters to pass to GenerationMixin.generate(). (See
            https://huggingface.co/docs/transformers/main_classes/text_generation)
        """
        if decode_kwargs is None:
            decode_kwargs = dict(skip_special_tokens=True)

        prompt = self.build_prompt(messages, functions)
        input_toks, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)

        # run it through the model
        output = self.model.generate(input_toks, **hyperparams)
        # decode to tokens
        # the completion shouldn't include the prompt or stop token
        content = self.tokenizer.decode(output[0][input_len:], **decode_kwargs).strip()
        # attempt to estimate the assistant message padding if not set
        output_len = len(output[0]) - (input_len + 1)
        self._chat_template_estimate_padding(content=content, n_tokens_generated=output_len, role=ChatRole.ASSISTANT)
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

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param streamer_timeout: The maximum number of seconds to wait for the next token when streaming.
        :param decode_kwargs: Any arguments to pass to AutoTokenizer.decode(). Defaults to
            ``dict(skip_special_tokens=True)``.
        :param hyperparams: Any additional parameters to pass to GenerationMixin.generate(). (See
            https://huggingface.co/docs/transformers/main_classes/text_generation)
        """
        if decode_kwargs is None:
            decode_kwargs = dict(skip_special_tokens=True)

        prompt = self.build_prompt(messages, functions)
        input_toks, input_len, hyperparams = self._get_generate_args(prompt, **hyperparams)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=streamer_timeout, **decode_kwargs)

        # run it through the model in another thread so that we can get the tokens in this thread
        output_toks = None

        def thread_target():
            nonlocal output_toks  # ugly way of sending the results of .generate to the outer scope
            output_toks = self.model.generate(input_toks, streamer=streamer, **hyperparams)

        thread = Thread(target=thread_target)
        thread.start()

        # then wait for tokens from the task
        yielded_tokens = []
        for token in streamer:
            yield token
            yielded_tokens.append(token)

        # clean up the thread
        thread.join()

        # yield a completion with usage stats
        content = "".join(yielded_tokens)
        # attempt to estimate the assistant message padding if not set
        output_len = len(yielded_tokens)
        self._chat_template_estimate_padding(content=content, n_tokens_generated=output_len, role=ChatRole.ASSISTANT)
        yield Completion(
            message=ChatMessage.assistant(content=content.strip()),
            prompt_tokens=input_len,
            completion_tokens=len(output_toks[0]) - (input_len + 1),
        )

    def _chat_template_estimate_padding(self, content: str, n_tokens_generated: int, role: ChatRole):
        """Estimate the number of padding tokens needed for"""
        if self.pipeline or self._padding_len_by_role[role]:
            return
        log.debug(f"Estimating {role} token padding from chat template...")
        reencoded_len = len(self.tokenizer.encode(content, skip_special_tokens=True))
        self._padding_len_by_role[role] = max(n_tokens_generated - reencoded_len, 0)
        log.debug(f"{n_tokens_generated=}, {reencoded_len=}, padding estimate={n_tokens_generated - reencoded_len}")


def _ensure_chat_template(tokenizer):
    if not hasattr(tokenizer, "apply_chat_template"):
        raise MissingModelDependencies(
            "To use the HuggingEngine with built-in chat templates requires `transformers>=4.34.0`. You currently"
            f" have `transformers=={transformers.__version__}`. Please update your transformers with `pip install"
            " -U transformers` or supply a `prompt_template` to this HuggingEngine."
        )
