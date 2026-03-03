import functools
import inspect
import itertools
import logging
import warnings
from typing import AsyncIterable, Literal

from kani import _optional
from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole
from . import function_calling, mm_tokens
from .model_constants import API_BY_PREFIX, CONTEXT_SIZES_BY_PREFIX
from .translation import ChatCompletion, OPENAI_PIPELINE, kani_cm_to_openai_cm, openai_tc_to_kani_tc
from .translation_responses import kani_cm_to_openai_responses_inputs, openai_responses_response_to_kani_completion
from .utils import DottableDict
from ..base import BaseCompletion, BaseEngine, Completion
from ..mixins import TokenCached

try:
    import tiktoken
    from openai import AsyncOpenAI as OpenAIClient
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.shared_params import FunctionDefinition
    from openai.types.responses import ResponseInputItemParam, ResponseInputParam
except ImportError as e:
    raise MissingModelDependencies(
        'The OpenAIEngine requires extra dependencies. Please install kani with "pip install kani[openai]".'
    ) from None

log = logging.getLogger(__name__)


class OpenAIEngine(TokenCached, BaseEngine):
    """
    Engine for using the OpenAI API.

    This engine supports all chat-based models and fine-tunes.

    **Multimodal support**: images, audio.

    **Message Extras**

    * ``"openai_completion"``: The ChatCompletion (raw response) returned by the OpenAI servers, as a dictionary.
      Non-streaming responses only.
    * ``"openai_usage"``: The usage data (raw response) returned by the OpenAI servers, as a dictionary.
    """

    disable_function_calling_kwargs = {"tool_choice": "none"}

    def __init__(
        self,
        api_key: str = None,
        model="gpt-4.1-nano",
        max_context_size: int = None,
        *,
        api_type: Literal["chat_completions", "responses"] = None,
        organization: str = None,
        retry: int = 5,
        api_base: str = "https://api.openai.com/v1",
        headers: dict = None,
        client: OpenAIClient = None,
        tokenizer=None,
        **hyperparams,
    ):
        """
        :param api_key: Your OpenAI API key. By default, the API key will be read from the `OPENAI_API_KEY` environment
            variable.
        :param model: The id of the model to use (e.g. "gpt-4o-mini", "ft:gpt-3.5-turbo:my-org:custom_suffix:id").
        :param max_context_size: The maximum amount of tokens allowed in the chat prompt. If None, uses the given
            model's full context size.
        :param api_type: Whether to use the Chat Completions API (default for most models) or Responses API (default for
            "deep-reasoning" style models). If unset, the best API type for the given model will be chosen.
        :param organization: The OpenAI organization to use in requests. By default, the org ID would be read from the
            `OPENAI_ORG_ID` environment variable (defaults to the API key's default org if not set).
        :param retry: How many times the engine should retry failed HTTP calls with exponential backoff (default 5).
        :param api_base: The base URL of the OpenAI API to use.
        :param headers: A dict of HTTP headers to include with each request.
        :param client: An instance of `openai.AsyncOpenAI <https://github.com/openai/openai-python>`_
            (for reusing the same client in multiple engines).
            You must specify exactly one of ``(api_key, client)``. If this is passed the ``organization``, ``retry``,
            ``api_base``, and ``headers`` params will be ignored.
        :param tokenizer: The tokenizer to use for token estimation - for OpenAI models this will be loaded
            automatically. A class with a ``.encode(text: str)`` method that returns a list (usually of token ids).
        :param hyperparams: The arguments to pass to the ``create_chat_completion`` call with each request. See
            https://platform.openai.com/docs/api-reference/chat/create for a full list of params.
        """
        if api_key and client:
            raise ValueError("You must supply no more than one of (api_key, client).")
        if max_context_size is None:
            matched_prefix, max_context_size = next(
                (prefix, size) for prefix, size in CONTEXT_SIZES_BY_PREFIX if model.startswith(prefix)
            )
            if not matched_prefix:
                warnings.warn(
                    "The context length for this model was not found, defaulting to 2048 tokens. Please specify"
                    " `max_context_size` if this is incorrect.",
                    stacklevel=2,
                )
        if api_type is None:
            matched_prefix, api_type = next((prefix, a) for prefix, a in API_BY_PREFIX if model.startswith(prefix))
            warnings.warn(
                f"The OpenAI API type for this model was not set, defaulting to {api_type!r} for {model!r}. "
                'Please specify `api_type="chat_completions"` or `api_type="responses"` if this is incorrect.',
                stacklevel=2,
            )

        super().__init__()

        self.client = client or OpenAIClient(
            api_key=api_key, organization=organization, max_retries=retry, base_url=api_base, default_headers=headers
        )
        self.model = model
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams
        self.openai_api_type = api_type
        self.tokenizer = tokenizer  # tiktoken caches a tokenizer globally in module, so we can unconditionally load it
        self._load_tokenizer()

    # ==== token counting ====
    def _load_tokenizer(self):
        if self.tokenizer:
            return
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            warnings.warn(
                f"Could not find a tokenizer for the {self.model} model. You may need to update tiktoken. Using"
                " o200k_base tokenizer as default."
            )
            self.tokenizer = tiktoken.get_encoding("o200k_base")

    def message_len(self, message: ChatMessage) -> int:
        if (cached_len := self.get_cached_message_len(message)) is not None:
            return cached_len

        mlen = 7
        # main content
        if _optional.has_multimodal_core:
            for part in message.parts:
                if isinstance(part, _optional.multimodal_core.AudioPart):
                    mlen += mm_tokens.tokens_from_audio_duration(part.duration, self.model)
                elif isinstance(part, _optional.multimodal_core.ImagePart):
                    mlen += mm_tokens.tokens_from_image_size(part.size, self.model)
                else:
                    mlen += len(self.tokenizer.encode(str(part)))
        else:
            if message.text:
                mlen += len(self.tokenizer.encode(message.text))

        # additional keys
        if message.name:
            mlen += len(self.tokenizer.encode(message.name))
        if message.tool_calls:
            for tc in message.tool_calls:
                mlen += len(self.tokenizer.encode(tc.function.name))
                mlen += len(self.tokenizer.encode(tc.function.arguments))

        # HACK: using gpt-4o and parallel function calling, the API randomly adds tokens based on the length of the
        # TOOL message (see tokencounting.ipynb)???
        # this seems to be ~ 6 + (token len / 20) tokens per message (though it randomly varies), but this formula
        # is <10 tokens of an overestimate in most cases
        if self.model.startswith("gpt-4o") and message.role == ChatRole.FUNCTION:
            mlen += 6 + (mlen // 20)

        self.set_cached_message_len(message, mlen)
        return mlen

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0
        # wrap an inner impl to use lru_cache with tuple
        return self._function_token_reserve_impl(tuple(functions))

    @functools.lru_cache(maxsize=256)
    def _function_token_reserve_impl(self, functions):
        prompt = function_calling.prompt(self.translate_functions(functions))
        return len(self.tokenizer.encode(prompt))

    @functools.cached_property
    def _count_tokens_arg_names(self):
        """A list of valid kwarg names that can be passed to self.client.responses.input_tokens.count"""
        try:
            inspected_params = set(inspect.signature(self.client.responses.input_tokens.count).parameters)
            return inspected_params
        except Exception as e:
            log.warning(
                "Could not introspect responses.input_tokens.count for parameter names, returning default:", exc_info=e
            )
            # default
            return {
                "conversation",
                "input",
                "instructions",
                "model",
                "parallel_tool_calls",
                "previous_response_id",
                "reasoning",
                "text",
                "tool_choice",
                "tools",
                "truncation",
                "extra_headers",
                "extra_query",
                "extra_body",
                "timeout",
            }

    # ==== hackable stuff for requests ====
    # --- kani -> oai translation ---
    def translate_functions(self, functions: list[AIFunction]) -> list[dict]:
        r"""Translate a list of Kani :class:`.AIFunction`\ s to a list of OpenAI tool definitions."""
        if self.openai_api_type == "chat_completions":
            return [
                dict(
                    type="function",
                    function=FunctionDefinition(name=f.name, description=f.desc, parameters=f.json_schema),
                )
                for f in functions
            ]
        elif self.openai_api_type == "responses":
            return [dict(type="function", name=f.name, description=f.desc, parameters=f.json_schema) for f in functions]
        raise ValueError(f"Unknown OpenAI API type: {self.openai_api_type!r}")

    def translate_messages(self, messages: list[ChatMessage]) -> list[ChatCompletionMessageParam] | ResponseInputParam:
        r"""Translate a list of Kani :class:`.ChatMessage`\ s to a list of OpenAI messages."""
        # we don't use a .apply() step here for hackability, so the pipeline just binds tool calls and cleans up
        # any invalid prefixes
        inter = OPENAI_PIPELINE(messages)
        if self.openai_api_type == "chat_completions":
            return [self.translate_kani_message_to_openai(m) for m in inter]
        elif self.openai_api_type == "responses":
            return list(
                itertools.chain.from_iterable(self.translate_kani_message_to_openai_responses(m) for m in inter)
            )
        raise ValueError(f"Unknown OpenAI API type: {self.openai_api_type!r}")

    def _prepare_request(
        self, messages, functions, *, intent: str = "chat_completions.create", **kwargs
    ) -> tuple[dict, list, dict | None]:
        """
        Prepare the API request to the OpenAI API. Returns a tuple (kwargs, messages, tools) to be passed to the
        OpenAIClient's chat.completions.create() or responses.create() method.

        :param messages: The Kani ChatMessages to translate into OpenAI-format messages.
        :param functions: The Kani AIFunctions to translate into OpenAI-format tools.
        :param intent: one of ("chat_completions.create", "chat_completions.stream", "responses.create",
            "responses.stream", "responses.input_tokens.count") -- the underlying OpenAI SDK call the returned keyword
            arguments will be passed to.
        :param kwargs: The request-specific kwargs passed to the request, from either the engine initialization or the
            chat_round call.
        """
        if functions:
            tool_specs = self.translate_functions(functions)
        else:
            tool_specs = None
        # translate to openai spec - group any tool messages together and ensure all free ToolCall IDs are bound
        translated_messages = self.translate_messages(messages)

        # responses API params
        if self.openai_api_type == "responses":
            # ensure include reasoning is set
            kwargs.setdefault("include", [])
            if "reasoning.encrypted_content" not in kwargs["include"]:
                kwargs["include"].append("reasoning.encrypted_content")

        return kwargs, translated_messages, tool_specs

    # --- oai -> kani translation ---
    def _translate_openai_chat_completion(self, completion):
        """Translate an OpenAI completion to a Kani completion. Only called for non-streaming requests by default."""
        if self.openai_api_type == "chat_completions":
            return ChatCompletion(openai_completion=completion)
        elif self.openai_api_type == "responses":
            return openai_responses_response_to_kani_completion(completion)
        raise ValueError(f"Unknown OpenAI API type: {self.openai_api_type!r}")

    # ==== chat completions ====
    @staticmethod
    def translate_kani_message_to_openai(message: ChatMessage) -> ChatCompletionMessageParam:
        """Translate a single Kani :class:`.ChatMessage` to a single OpenAI message."""
        return kani_cm_to_openai_cm(message)

    async def _predict_chat_completions(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> ChatCompletion:
        local_kwargs, translated_messages, tool_specs = self._prepare_request(
            messages, functions, intent="chat_completions.create", **(self.hyperparams | hyperparams)
        )
        # make API call
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=translated_messages,
            tools=tool_specs,
            **local_kwargs,
        )
        # translate into Kani spec and return
        kani_cmpl = self._translate_openai_chat_completion(completion)
        self.set_cached_prompt_len(messages, functions, completion.usage.prompt_tokens)
        self.set_cached_prompt_len(
            messages + [kani_cmpl.message], functions, completion.usage.prompt_tokens + kani_cmpl.completion_tokens
        )
        self.set_cached_message_len(kani_cmpl.message, kani_cmpl.completion_tokens)
        return kani_cmpl

    async def _stream_chat_completions(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        local_kwargs, translated_messages, tool_specs = self._prepare_request(
            messages, functions, intent="chat_completions.stream", **(self.hyperparams | hyperparams)
        )
        # make API call
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=translated_messages,
            tools=tool_specs,
            stream=True,
            stream_options={"include_usage": True},
            **local_kwargs,
        )

        # save requested tool calls and content as streamed
        content_chunks = []
        tool_call_partials = {}  # index -> tool call
        usage = None

        # iterate over the stream and yield/save
        async for chunk in stream:
            # save usage if present
            if chunk.usage is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            # process content delta
            delta = chunk.choices[0].delta

            # yield content
            if delta.content is not None:
                content_chunks.append(delta.content)
                yield delta.content

            # tool calls are partials, save a mapping to the latest state and we'll translate them later once complete
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    if tc_delta.index not in tool_call_partials:
                        tool_call_partials[tc_delta.index] = tc_delta
                    else:
                        partial = tool_call_partials[tc_delta.index]
                        if tc_delta.function.name is not None:
                            partial.function.name += tc_delta.function.name
                        if tc_delta.function.arguments is not None:
                            partial.function.arguments += tc_delta.function.arguments

        # construct the final completion with streamed tool calls
        content = None if not content_chunks else "".join(content_chunks)
        tool_calls = [openai_tc_to_kani_tc(tc) for tc in sorted(tool_call_partials.values(), key=lambda c: c.index)]
        msg = ChatMessage(role=ChatRole.ASSISTANT, content=content, tool_calls=tool_calls)

        # token counting
        if usage:
            self.set_cached_prompt_len(messages, functions, usage.prompt_tokens)
            self.set_cached_prompt_len(messages + [msg], functions, usage.prompt_tokens + usage.completion_tokens)
            self.set_cached_message_len(msg, usage.completion_tokens)
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            msg.extra["openai_usage"] = DottableDict(usage.model_dump(mode="json"))
        else:
            prompt_tokens = completion_tokens = None
        yield Completion(message=msg, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    # ==== responses ====
    @staticmethod
    def translate_kani_message_to_openai_responses(message: ChatMessage) -> list[ResponseInputItemParam]:
        """Translate a single Kani :class:`.ChatMessage` to its corresponding OpenAI responses input items."""
        return kani_cm_to_openai_responses_inputs(message)

    async def _predict_responses(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> ChatCompletion:
        local_kwargs, translated_messages, tool_specs = self._prepare_request(
            messages, functions, intent="responses.create", **(self.hyperparams | hyperparams)
        )
        # make API call
        response = await self.client.responses.create(
            model=self.model, input=translated_messages, tools=tool_specs, **local_kwargs
        )
        # translate into Kani spec and return
        kani_cmpl = self._translate_openai_chat_completion(response)
        self.set_cached_prompt_len(messages, functions, response.usage.input_tokens)
        self.set_cached_prompt_len(messages + [kani_cmpl.message], functions, response.usage.total_tokens)
        self.set_cached_message_len(kani_cmpl.message, kani_cmpl.completion_tokens)
        return kani_cmpl

    async def _stream_responses(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        local_kwargs, translated_messages, tool_specs = self._prepare_request(
            messages, functions, intent="responses.stream", **(self.hyperparams | hyperparams)
        )
        # the sdk handles the text streaming for us, so we can just use that
        async with self.client.responses.stream(
            model=self.model, input=translated_messages, tools=tool_specs or [], **local_kwargs
        ) as streamer:
            async for event in streamer:
                # we only want to emit the content of response.output_text.delta
                if event.type == "response.output_text.delta":
                    yield event.delta

            # translate into Kani spec and return
            response = await streamer.get_final_response()
            kani_cmpl = self._translate_openai_chat_completion(response)
            self.set_cached_prompt_len(messages, functions, response.usage.input_tokens)
            self.set_cached_prompt_len(messages + [kani_cmpl.message], functions, response.usage.total_tokens)
            self.set_cached_message_len(kani_cmpl.message, kani_cmpl.completion_tokens)
            yield kani_cmpl

    # ==== main kani impl ====
    async def prompt_len(self, messages, functions=None, **kwargs) -> int:
        # we have a token counting endpoint for responses api, so prepare and count
        if self.openai_api_type == "responses":
            local_kwargs, translated_messages, tool_specs = self._prepare_request(
                messages, functions, intent="responses.input_tokens.count", **(self.hyperparams | kwargs)
            )
            valid_count_token_kwargs = {k: v for k, v in local_kwargs.items() if k in self._count_tokens_arg_names}
            resp = await self.client.responses.input_tokens.count(
                model=self.model, input=translated_messages, tools=tool_specs, **valid_count_token_kwargs
            )
            return resp.input_tokens

        # for chat completions api, we have to count ourselves
        # optimization: since chat-based appends messages 1 at a time, see if the messages - 1 is in prompt cache
        if messages and (cached := self.get_cached_prompt_len(messages[:-1], functions, **kwargs)) is not None:
            return cached + self.message_len(messages[-1])
        # OpenAI does not use an API for token counting, so we'll use the old-style message-wise counting
        return sum(self.message_len(m) for m in messages) + self.function_token_reserve(functions)

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> ChatCompletion:
        if self.openai_api_type == "chat_completions":
            return await self._predict_chat_completions(messages, functions, **hyperparams)
        elif self.openai_api_type == "responses":
            return await self._predict_responses(messages, functions, **hyperparams)
        raise ValueError(f"Unknown OpenAI API type: {self.openai_api_type!r}")

    async def stream(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        if self.openai_api_type == "chat_completions":
            async for item in self._stream_chat_completions(messages, functions, **hyperparams):
                yield item
        elif self.openai_api_type == "responses":
            async for item in self._stream_responses(messages, functions, **hyperparams):
                yield item
        else:
            raise ValueError(f"Unknown OpenAI API type: {self.openai_api_type!r}")

    async def close(self):
        await self.client.close()

    def __repr__(self):
        return (
            f"{type(self).__name__}(model={self.model}, max_context_size={self.max_context_size},"
            f" hyperparams={self.hyperparams})"
        )
