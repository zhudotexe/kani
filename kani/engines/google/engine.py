import asyncio
import datetime
import functools
import io
import json
import logging
import os
import warnings
from typing import AsyncIterable

from kani import _optional
from kani.ai_function import AIFunction
from kani.exceptions import MissingModelDependencies
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall
from kani.prompts.pipeline import PromptPipeline
from . import mm_tokens, model_constants
from ..base import BaseCompletion, BaseEngine, Completion
from ..mixins import TokenCached

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as e:
    raise MissingModelDependencies(
        'The GoogleAIEngine requires extra dependencies. Please install kani with "pip install kani[google]".'
    ) from None


log = logging.getLogger(__name__)


# ==== pipe ====
# assumes system messages are plucked before calling
GOOGLE_PIPELINE: PromptPipeline[list[genai_types.Content]] = (
    PromptPipeline().translate_role(role=ChatRole.SYSTEM, to=ChatRole.USER).ensure_bound_function_calls()
)
ROLE_TRANSFORMS = {
    ChatRole.ASSISTANT: "model",
    ChatRole.FUNCTION: "tool",
}


class GoogleAIEngine(TokenCached, BaseEngine):
    """
    Engine for using the Google AI Studio API (aka Gemini Developer API, Google AI API) and
    Google Vertex AI API (aka Google Cloud API).

    This engine supports all Google AI models.

    See https://ai.google.dev/gemini-api/docs/models for a list of available models.

    **Multimodal support**: images, audio, video.

    **Message Extras**: ``"google_response"``: The
    `raw response <https://ai.google.dev/api/generate-content#generatecontentresponse>`_ returned by the Google AI API.
    """

    # because we have to estimate tokens wildly and the ctx is so long we'll just reserve a bunch
    token_reserve = 500

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.5-flash",
        max_context_size: int = None,
        *,
        # client settings
        retry: int = 2,
        api_base: str = None,
        headers: dict = None,
        client: genai.Client = None,
        # kani settings
        multimodal_upload_bytes_threshold: int = 512_000,
        **hyperparams,
    ):
        """
        :param api_key: Your Gemini Developer API key. By default, the API key will be read from the `GEMINI_API_KEY`
            environment variable.
        :param model: The id of the model to use (e.g. "gemini-2.5-flash"). See
            https://ai.google.dev/gemini-api/docs/models for a list of models.
        :param max_tokens: The maximum number of tokens to sample at each generation (defaults to 512).
            Generally, you should set this to the same number as your Kani's ``desired_response_tokens``.
        :param max_context_size: The maximum amount of tokens allowed in the chat prompt. If None, uses the given
            model's full context size.
        :param retry: How many times the engine should retry failed HTTP calls with exponential backoff (default 2).
        :param api_base: The base URL of the Google AI API to use. If not specified, the default URL for the specified
            API (AI Studio/Vertex) will be used.
        :param headers: A dict of HTTP headers to include with each request.
        :param client: An instance of ``genai.Client`` (for reusing the same client in multiple engines).
            You must specify exactly one of (api_key, client).
        :param multimodal_upload_bytes_threshold: If a multimodal object (audio, image, video) is larger than this
            number of bytes, upload it as a file instead of passing it inline in a request. Default 512kB.
        :param hyperparams: Any additional parameters to pass to the underlying API call (see
            https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig).
        """
        if api_key and client:
            raise ValueError("You must supply no more than one of (api_key, client).")
        if api_key is None and client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "You must supply an `api_key`, `client`, or set the `GEMINI_API_KEY` environment variable to use"
                    " the GoogleAIEngine."
                )
        if max_context_size is None:
            matched_prefix, max_context_size = next(
                (prefix, size) for prefix, size in model_constants.CONTEXT_SIZES_BY_PREFIX if model.startswith(prefix)
            )
            if not matched_prefix:
                warnings.warn(
                    f"The context length for this model was not found, defaulting to {max_context_size} tokens. Please"
                    " specify `max_context_size` if this is incorrect."
                )

        super().__init__()

        self.client = client or genai.Client(
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=retry),
                base_url=api_base,
                headers=headers,
            )
        )
        self.model = model
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams

        # multimodal file caching
        self.multimodal_upload_bytes_threshold = multimodal_upload_bytes_threshold
        self._multimodal_file_cache: dict[bytes, genai.types.File] = {}  # multimodal part sha256 -> google file

    # ==== token counting ====
    def message_len(self, message: ChatMessage) -> int:
        if (cached_len := self.get_cached_message_len(message)) is not None:
            return cached_len

        # TODO with async token counting use the token counting API
        chars = len(message.role.value)
        tokens = 0
        if _optional.has_multimodal_core:
            for part in message.parts:
                if isinstance(part, _optional.multimodal_core.ImagePart):
                    tokens += mm_tokens.tokens_from_image_size(part.size, self.model)
                elif isinstance(part, _optional.multimodal_core.AudioPart):
                    tokens += mm_tokens.tokens_from_audio_duration(part.duration, self.model)
                elif isinstance(part, _optional.multimodal_core.VideoPart):
                    tokens += mm_tokens.tokens_from_video_duration(part.duration, self.model)
                else:
                    chars += len(str(part))
        else:
            chars += len(message.text)

        # tools
        if message.tool_calls:
            for tc in message.tool_calls:
                chars += len(tc.function.name) + len(tc.function.arguments)

        # Google documents 4 bytes per token, so we do a conservative 3.8 char/tok
        return int(chars / 3.8) + tokens

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0
        # wrap an inner impl to use lru_cache with frozensets
        return self._function_token_reserve_impl(frozenset(functions))

    @functools.lru_cache(maxsize=256)
    def _function_token_reserve_impl(self, functions):
        # panik, also assume len/4?
        n = sum(len(f.name) + len(f.desc) + len(json.dumps(f.json_schema)) for f in functions)
        return int(n / 3.8)

    # ==== requests ====
    async def _prepare_request(
        self, messages, functions, hyperparams
    ) -> tuple[genai_types.GenerateContentConfigDict, list[genai_types.Content]]:
        """
        Prepare the API request to the Google AI API. Returns a tuple (GenerateContentConfigDict, Content[]) to be
        passed to the genai Client's ``generate_content()`` method.
        """
        kwargs = {}

        # --- messages ---
        # pluck system messages
        last_system_idx = next((i for i, m in enumerate(messages) if m.role != ChatRole.SYSTEM), None)
        if last_system_idx:
            kwargs["system_instruction"] = "\n\n".join(m.text for m in messages[:last_system_idx])
            messages = messages[last_system_idx:]

        # enforce ordering and function call bindings
        translated_messages = GOOGLE_PIPELINE(messages)

        # translate to content list
        translated_messages = [await self._translate_message(m) for m in translated_messages]

        # --- tools ---
        if functions:
            kwargs.setdefault("tools", [])
            kwargs["tools"].append(
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(
                            name=f.name,
                            description=f.desc,
                            parameters=genai_types.Schema.from_json_schema(
                                json_schema=genai_types.JSONSchema.model_validate(f.json_schema)
                            ),
                        )
                        for f in functions
                    ]
                )
            )

        # --- kwargs ---
        kwargs.update(self.hyperparams)
        kwargs.update(hyperparams)

        log.debug(f"translated prompt: {translated_messages}")

        return kwargs, translated_messages

    async def _translate_message(self, msg: ChatMessage) -> genai_types.Content:
        """
        Translate one message into a Content object;
        automatically upload and save references to large multimodal objects using Files API (max 2GB per file,
        20GB total, autodeletes after 48h).
        """
        role = ROLE_TRANSFORMS.get(msg.role, msg.role.value)
        content = []

        # FUNCTION
        if msg.role == ChatRole.FUNCTION:
            # tool call error
            if msg.is_tool_call_error:
                content.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            id=msg.tool_call_id, name=msg.name, response={"error": msg.text}
                        )
                    )
                )
            else:
                content.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            id=msg.tool_call_id, name=msg.name, response={"result": msg.text}
                        )
                    )
                )
        # ASSISTANT, USER messages
        else:
            for part in msg.parts:
                # --- multimodal ---
                if _optional.has_multimodal_core and isinstance(
                    part,
                    (
                        _optional.multimodal_core.ImagePart,
                        _optional.multimodal_core.AudioPart,
                        _optional.multimodal_core.BinaryFilePart,
                    ),
                ):
                    content.append(await self._translate_multimodal_part(part))
                # default
                else:
                    content.append(genai_types.Part(text=str(part)))

        # ASSISTANT messages with tool calls
        if msg.role == ChatRole.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                content.append(
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(id=tc.id, name=tc.function.name, args=tc.function.kwargs)
                    )
                )

        return genai_types.Content(role=role, parts=content)

    async def _translate_multimodal_part(self, part) -> genai_types.Part:
        """
        Translate a multimodal kani part to a google part, uploading it to the Files API if it's large.

        Caches uploaded files for re-use based on sha256.
        """
        # if we have uploaded this file to the files API before, add the file part
        sha256 = part.sha256()
        if sha256 in self._multimodal_file_cache:
            google_file = self._multimodal_file_cache[sha256]
            # check if the upload is still valid
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            if now < google_file.expiration_time:
                log.debug(f"Using cached google file part: {google_file}")
                return genai_types.Part.from_uri(file_uri=google_file.uri, mime_type=google_file.mime_type)
            log.debug(f"Google file part is expired, falling through to re-upload")
        # otherwise read the file
        # image
        if isinstance(part, _optional.multimodal_core.ImagePart):
            media_type = "image/png"
            data = part.as_bytes(format="png")
        # audio
        elif isinstance(part, _optional.multimodal_core.AudioPart):
            media_type = "audio/wav"
            data = part.as_wav_bytes()
        # video/arbitrary binary
        elif isinstance(part, _optional.multimodal_core.BinaryFilePart):
            media_type = part.mime
            data = part.as_bytes()
        else:
            raise ValueError(
                f"Invalid multimodal message part: {part!r}. This should never happen. Please open a"
                " bug report with reproduction steps."
            )

        # if the file data is more than the threshold, upload it and use the file part
        if len(data) >= self.multimodal_upload_bytes_threshold:
            log.debug(f"Uploading multimodal file to Files API (len={len(data)})")
            google_file = await self.client.aio.files.upload(
                file=io.BytesIO(data), config=genai_types.UploadFileConfig(mime_type=media_type)
            )
            log.debug(google_file)
            if google_file.state not in (genai_types.FileState.ACTIVE, genai_types.FileState.PROCESSING):
                raise RuntimeError(f"Invalid google file state, file: {google_file}")
            self._multimodal_file_cache[sha256] = google_file
            google_part = genai_types.Part.from_uri(file_uri=google_file.uri, mime_type=google_file.mime_type)
            if google_file.state == genai_types.FileState.ACTIVE:
                return google_part
            # wait until the file is done processing
            log.debug(f"Uploaded google file part is not ACTIVE, waiting for ACTIVE")
            for idx in range(50):
                # poll every 5s since google does not offer a blocking wait
                await asyncio.sleep(5)
                google_file = await self.client.aio.files.get(name=google_file.name)
                log.debug(f"{(idx + 1) * 5} sec...\n{google_file}")
                if google_file.state == genai_types.FileState.ACTIVE:
                    self._multimodal_file_cache[sha256] = google_file
                    return google_part
            self._multimodal_file_cache.pop(sha256, None)
            raise RuntimeError(
                f"Google file state is not ACTIVE after long wait, something might be wrong!\n{google_file}"
            )
        # otherwise just include the bytes inline
        return genai_types.Part.from_bytes(data=data, mime_type=media_type)

    def _translate_google_response(self, resp: genai_types.GenerateContentResponse) -> Completion:
        tool_calls = []
        parts = []
        for part in resp.candidates[0].content.parts:
            if part.text:
                parts.append(part.text)
            elif part.function_call:
                fc = FunctionCall(name=part.function_call.name, arguments=json.dumps(part.function_call.args))
                tc = ToolCall.from_function_call(fc, call_id_=part.function_call.id)
                tool_calls.append(tc)
            else:
                warnings.warn(
                    f"The engine returned an unknown part: {part}. This will not be returned in the ChatMessage. To"
                    ' access this part, use `message.extra["google_response"].candidates[0].content.parts`.'
                )

        if len(parts) == 1:
            content = parts[0]
        elif not parts:
            content = None
        else:
            content = parts

        kani_msg = ChatMessage.assistant(content, tool_calls=tool_calls or None)

        # also cache the message token len
        self.set_cached_message_len(kani_msg, resp.usage_metadata.candidates_token_count)

        # set the extra
        kani_msg.extra["google_response"] = resp
        return Completion(
            message=kani_msg,
            prompt_tokens=resp.usage_metadata.prompt_token_count,
            completion_tokens=resp.usage_metadata.candidates_token_count,
        )

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> Completion:
        request_config, prompt_msgs = await self._prepare_request(messages, functions, hyperparams)

        # --- completion ---
        assert len(prompt_msgs) > 0
        message = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt_msgs,
            config=request_config,
        )

        # translate to kani
        return self._translate_google_response(message)

    async def stream(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        # do the stream
        request_config, prompt_msgs = await self._prepare_request(messages, functions, hyperparams)

        assert len(prompt_msgs) > 0
        last_chunk = None
        content_parts = []
        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=prompt_msgs,
            config=request_config,
        ):
            for part in chunk.candidates[0].content.parts:
                if part.text:
                    yield part.text
            last_chunk = chunk
            content_parts.extend(chunk.candidates[0].content.parts)

        if last_chunk:
            last_chunk.candidates[0].content.parts = content_parts
            yield self._translate_google_response(last_chunk)
