import logging
import re
import warnings
from typing import AsyncIterable

from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage
from kani.prompts.pipeline import PromptPipeline
from ..base import BaseCompletion, BaseEngine, Completion

try:
    import torch
    from llama_cpp import Llama
except ImportError:
    raise MissingModelDependencies(
        'The LlamaCppEngine requires extra dependencies. Please install kani with "pip install kani[cpp]". '
    ) from None

log = logging.getLogger(__name__)


class LlamaCppEngine(BaseEngine):
    """
    This class implements the main decoding logic for any GGUF model (not just LLaMA as the name might suggest).

    **GPU Support**

    llama.cpp supports multiple acceleration backends, which may require different flags to be set during installation.
    To see the full list of backends, see their README at https://github.com/abetlen/llama-cpp-python.

    To load some or all of the model layers on GPU, pass ``n_gpu_layers=...`` in the ``model_load_kwargs``. Use
    ``-1`` to specify all layers.
    """

    def __init__(
        self,
        repo_id: str,
        filename: str = None,
        max_context_size: int = 0,
        prompt_pipeline: PromptPipeline[str | list[int]] = None,
        *,
        model_load_kwargs: dict = None,
        **hyperparams,
    ):
        """
        :param repo_id: The ID of the model repo to load from Hugging Face.
        :param filename: A filename or glob pattern to match the model file in the repo.
        :param max_context_size: The context size of the model.
        :param prompt_pipeline: The pipeline to translate a list of kani ChatMessages into the model-specific chat
            format (see :class:`.PromptPipeline`).
        :param model_load_kwargs: Additional arguments to pass to ``Llama.from_pretrained()``.
            See `this link <https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.from_pretrained>`_
            for more info.
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        if model_load_kwargs is None:
            model_load_kwargs = {}

        self.repo_id = repo_id
        self.filename = filename
        self.pipeline = prompt_pipeline

        # for convenience, if the filename is *-00001-of-0000X.gguf, mark all the others as additional files if not set
        if match := re.match(r"(.*?)-(\d+)-of-(\d+)\.gguf", filename):
            log.info("Sharded GGUF file given - ensuring that all GGUF shards are downloaded")
            # there is an issue in llama-cpp-python that makes the additional_files inherit the subfolder of the parent
            # https://github.com/abetlen/llama-cpp-python/issues/1938
            if "/" in match[1]:
                warnings.warn(
                    "llama-cpp-python can fail to find additional model files in subfolders. If you see a 404 error,"
                    " try manually using huggingface-cli to download model files. See"
                    " https://github.com/abetlen/llama-cpp-python/issues/1938 for more information."
                )
            additional_files = []
            for n in range(1, int(match[3]) + 1):
                if n == int(match[2]):
                    continue
                additional_files.append(f"{match[1]}-*{n}-of-{match[3]}.gguf")
            log.info(f"additional_files={additional_files}")
            model_load_kwargs.setdefault("additional_files", additional_files)

        model_load_kwargs.setdefault("n_ctx", max_context_size)
        self.model = Llama.from_pretrained(repo_id=repo_id, filename=filename, **model_load_kwargs)
        self.hyperparams = hyperparams

        self.max_context_size = max_context_size or self.model.n_ctx()

        if self.token_reserve == 0 and self.pipeline:
            self.token_reserve = self._infer_token_reserve()

    def _infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        prompt = self.pipeline.execute([], for_measurement=True)
        if isinstance(prompt, list):
            return len(prompt)
        tokenized = self.model.tokenize(prompt.encode(), add_bos=False, special=True)
        return len(tokenized)

    def message_len(self, message: ChatMessage) -> int:
        # default concrete base behaviour:
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the LlamaCppEngine to use it as a non-abstract class. If your model"
                " uses a chat template (or is a quantization of a model with a chat template), you can use the"
                " following:\n"
                "from kani.engines.huggingface import ChatTemplatePromptPipeline\n"
                "pipeline = ChatTemplatePromptPipeline.from_pretrained(base_model_id)\n"
                "engine = LlamaCppEngine(..., prompt_pipeline=pipeline)"
            )
        prompt = self.pipeline.execute([message], for_measurement=True)
        if isinstance(prompt, list):
            return len(prompt)
        elif isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        tokenized = self.model.tokenize(prompt.encode(), add_bos=False, special=True)
        return len(tokenized)

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0
        # default concrete base behaviour:
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the LlamaCppEngine to use it as a non-abstract class. If your model"
                " uses a chat template (or is a quantization of a model with a chat template), you can use the"
                " following:\n"
                "from kani.engines.huggingface import ChatTemplatePromptPipeline\n"
                "pipeline = ChatTemplatePromptPipeline.from_pretrained(base_model_id)\n"
                "engine = LlamaCppEngine(..., prompt_pipeline=pipeline)"
            )
        prompt = self.pipeline.execute([], functions, for_measurement=True)
        if isinstance(prompt, list):
            return len(prompt)
        elif isinstance(prompt, torch.Tensor):
            toklen = len(prompt[0])
        else:
            # prompt str to tokens
            tokenized = self.model.tokenize(prompt.encode(), add_bos=False, special=False)
            toklen = len(tokenized)

        # warn if there are functions but no tokens
        if toklen == 0:
            warnings.warn(
                "Functions were given to the model, but the function prompt returned 0 tokens! This model may not"
                " support function calling, or you may need to implement"
                f" `{type(self).__name__}.function_token_reserve()`."
            )

        return toklen

    def build_prompt(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None) -> str | list[int]:
        """
        Given the list of messages from kani, build either a single string representing the prompt for the model,
        or build the token list.

        The default behaviour is to call the supplied pipeline.
        """
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the LlamaCppEngine to use it as a non-abstract class. If your model"
                " uses a chat template (or is a quantization of a model with a chat template), you can use the"
                " following:\n"
                "from kani.engines.huggingface import ChatTemplatePromptPipeline\n"
                "pipeline = ChatTemplatePromptPipeline.from_pretrained(base_model_id)\n"
                "engine = LlamaCppEngine(..., prompt_pipeline=pipeline)"
            )
        prompt = self.pipeline(messages, functions)
        log.debug(f"BUILT PROMPT: {prompt}")
        return prompt

    def _get_generate_args(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams):
        """Internal method to build common params for the generate call"""
        prompt = self.build_prompt(messages, functions)
        if isinstance(prompt, str):
            # prompt str to tokens
            input_toks = self.model.tokenize(prompt.encode(), add_bos=False, special=True)
            input_len = len(input_toks)
        elif isinstance(prompt, list):
            input_toks = prompt
            input_len = len(input_toks)
        else:
            raise TypeError("build_prompt should either return a str or a list[int].")
        # set up hyperparams
        hyperparams = {**self.hyperparams, **hyperparams}
        hyperparams.setdefault("max_tokens", None)  # by default llama.cpp sets this to 16, which is too small
        return input_toks, input_len, hyperparams

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param hyperparams: Any additional parameters to pass to ``Llama.create_completion()``. (See
            https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion)
        """
        input_toks, input_len, hyperparams = self._get_generate_args(messages, functions, **hyperparams)

        completion = self.model.create_completion(input_toks, **hyperparams)
        return Completion(
            ChatMessage.assistant(completion["choices"][0]["text"]),
            prompt_tokens=input_len,
            completion_tokens=completion["usage"]["completion_tokens"],
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        **hyperparams,
    ) -> AsyncIterable[str | BaseCompletion]:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param hyperparams: Any additional parameters to pass to ``Llama.create_completion()``. (See
            https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion)
        """
        input_toks, input_len, hyperparams = self._get_generate_args(messages, functions, **hyperparams)

        stream = self.model.create_completion(input_toks, stream=True, **hyperparams)

        # iterate over the stream and yield/save
        content_chunks = []
        for chunk in stream:
            text = chunk["choices"][0]["text"]

            # yield content
            if text is not None:
                content_chunks.append(text)
                yield text

        # construct the final completion
        # https://github.com/abetlen/llama-cpp-python/issues/1498 blocks token counting impl
        content = None if not content_chunks else "".join(content_chunks)
        yield Completion(message=ChatMessage.assistant(content))
