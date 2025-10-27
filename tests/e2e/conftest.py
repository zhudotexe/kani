"""
E2E engines used in tests.

The engines will use the cache at _cache to return values from a hydrate run, based on the content of the request
to the engine. If a given request is not cached, fail the test.

If the ``KANI_E2E_HYDRATE=api,local`` env var is set, any missing cache entries for one of the E2E mock engines will
make a request to the upstream, and save it for future caching. Remove ``api`` or ``local`` to run a subset.
"""

import asyncio
import datetime
import hashlib
import json
import logging
import mimetypes
import os
import pprint
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from anthropic import AsyncAnthropic
from freezegun import freeze_time
from google import genai
from openai import AsyncOpenAI
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from kani import model_specific
from kani.engines.anthropic import AnthropicEngine
from kani.engines.google import GoogleAIEngine
from kani.engines.huggingface import HuggingEngine
from kani.engines.llamacpp import LlamaCppEngine
from kani.engines.openai import OpenAIEngine

# the directory where mock calls & returns are saved; should be committed to Git
MOCK_CACHE_BASE = Path(__file__).parent / "_cache"
DO_REAL_API_REQUESTS = "api" in os.getenv("KANI_E2E_HYDRATE", "")
DO_REAL_LOCAL_GENERATE = "local" in os.getenv("KANI_E2E_HYDRATE", "")
CI_TORCH_DEVICE = os.getenv("CI_TORCH_DEVICE")  # we want to force CPU in GHA

log = logging.getLogger("tests.e2e")


# ==== caching utils ====
# --- datetime ---
# certain prompt templates include the current date and time, we'll set it to an arbitrary fixed time
# (the day I wrote these tests)
@pytest.fixture(autouse=True, scope="session")
def mock_date():
    with freeze_time(
        datetime.datetime(2025, 10, 21, 12, 55, 37),
        real_asyncio=True,  # for asyncio internals
        tick=True,  # so pycharm reports correct test timing
    ):
        yield


# --- http ---
# headers whose values should be saved in the request
REQUEST_KEPT_HEADERS = {"content-type", "accept"}
# headers whose presence should be saved in the request but value redacted
REDACTED_HEADERS = {"authorization", "x-api-key", "x-goog-api-key"}
# headers which should be removed from the response
RESPONSE_REMOVED_HEADERS = {"content-encoding", "set-cookie"}


def cache_key_for_http_request(request: httpx.Request) -> str:
    """Get the cache key (filename) for a given HTTP request."""
    body = fmt_http_request(request)
    if request.headers["content-type"].startswith("application/json"):
        # HACK: for some reason, the exact binary a multimodal part gets translated to is platform-dependent
        # maybe some weirdness with PIL png formatting/ffmpeg across platforms
        # either way we'll just replace known image/audio dicts with a dummy for hashing, recursively
        head, content = body.split(b"\n\n", 1)
        # anthropic
        content = re.sub(
            r'"media_type":\s?"image/png",\s?"data":\s?"[0-9a-zA-Z/=]+"',
            '"media_type":"image/png","data":"dummy"',
            content.decode(),
        )
        # google
        content = re.sub(
            r'"data":\s?"[0-9a-zA-Z/=]+",\s?"mimeType":\s?"(image/png|audio/wav)"',
            r'"data": "dummy", "mimeType": "\1"',
            content,
        )
        # openai
        content = re.sub(
            r"data:image/png;base64,[0-9a-zA-Z/=]+",
            "data:image/png;base64,dummy",
            content,
        )
        body = head + b"\n\n" + content.encode()

    the_hash = hashlib.sha256(body)
    *_, last_path_segment = request.url.path.split("/")
    return f"{last_path_segment}-{the_hash.hexdigest()}"


def cache_dir_for_http_request(request: httpx.Request) -> Path:
    """Get a cache dir per hostname. I.e. _cache/<host>/<endpoint>_<hash>/"""
    cache_key = cache_key_for_http_request(request)
    cache_dir = MOCK_CACHE_BASE / request.url.host / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def fmt_http_request(request: httpx.Request) -> bytes:
    """
    Format an HTTP request in human-readable format.

    Redact any headers in the REDACTED_HEADERS list. Don't save any headers not in the KEPT_HEADERS list.
    Returns bytes in case our content is binary/for hashing later.
    """
    # head
    parts = [request.method.encode() + b" " + str(request.url).encode()]
    for k, v in request.headers.items():
        if k.lower() in REDACTED_HEADERS:
            parts.append(k.encode() + b": *** REDACTED ***")
        elif k.lower() in REQUEST_KEPT_HEADERS:
            parts.append(k.encode() + b": " + str(v).encode())
        else:
            log.debug(f"Discarding header: {k}: {v}")
    # content
    content = request.read()
    if content:
        parts.append(b"\n" + content)
    return b"\n".join(parts)


class AsyncCachingTransport(httpx.AsyncHTTPTransport):
    """
    A transport layer that hashes the request and returns a response if present, otherwise looks at the KANI_E2E_HYDRATE
    env var to determine whether to forward it upstream or err.

    This handles streaming too since streaming is just a long-lived bytestream in the response.
    """

    async def handle_async_request(self, request: httpx.Request):
        # find the cache file for the request
        cache_dir = cache_dir_for_http_request(request)
        req_path = cache_dir / "_request.http"
        resp_head_path = cache_dir / f"_head.json"

        # save the request to the cache dir
        req_path.write_bytes(fmt_http_request(request))

        # return the cached resp
        if resp_head_path.exists():
            try:
                with open(resp_head_path) as f:
                    meta = json.load(f)
                resp_body_path = resp_head_path.with_name(meta["body_path"])
                print(f"Returning cached response from: {cache_dir}")
                return httpx.Response(
                    status_code=meta["status"],
                    headers=httpx.Headers(meta["headers"]),
                    content=resp_body_path.read_bytes(),
                )
            except Exception as e:
                log.warning(
                    f"Could not load cached response from {resp_head_path}, falling back to make new request!",
                    exc_info=e,
                )

        # either pass it on or explode
        if DO_REAL_API_REQUESTS:
            print("Generating new response!")
            resp = await super().handle_async_request(request)
            headers = {k: v for k, v in resp.headers.items() if k.lower() not in RESPONSE_REMOVED_HEADERS}
            content_type, *_ = resp.headers["content-type"].split(";")

            # save the body to a filetype with the right extension for human readability
            ext = mimetypes.guess_extension(content_type)
            if ext:
                resp_body_path = cache_dir / f"body{ext}"
            else:
                resp_body_path = cache_dir / "body.bin"

            # save head to cache
            with open(resp_head_path, "w") as f:
                json.dump(
                    {"status": resp.status_code, "headers": headers, "body_path": resp_body_path.name},
                    f,
                    indent=2,
                )

            # save body to cache
            # we'll special-case JSON with indent=2 for human readability
            if ext == ".json":
                with open(resp_body_path, "w") as f:
                    await resp.aread()
                    json.dump(resp.json(), f, indent=2)
            else:
                # we can't stream it here since that reasonably consumes the stream without saving to ._content
                # so hopefully the model providers aren't returning GB+ responses
                resp_body_path.write_bytes(await resp.aread())
            print(f"New response saved to cache at: {cache_dir}")

            return resp
        raise ValueError(
            f"Request was not cached: {req_path}. This may mean the request has changed, or you need to hydrate the"
            " request cache. Use `KANI_E2E_HYDRATE=api pytest -m e2e` to hydrate request cache."
        )


# --- tokenwise ---
def cache_dir_for_local_generate(model_id: str, cache_key: str) -> Path:
    """Get a cache dir per model name. E.g. _cache/openai__gpt-oss-20b/"""
    cache_dir = MOCK_CACHE_BASE / model_id.replace("/", "__") / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_key_for_local_generate(input_ids: torch.Tensor, input_features: torch.Tensor = None) -> str:
    """Get the cache key (filename) for a given generate request."""
    the_hash = hashlib.sha256()
    # torch.save saves a pickle, so we want to hash the contents of the tensor ourselves
    # we do this lazily by just hashing the repr of the tensor
    the_hash.update(str(input_ids.tolist()).encode())
    # the features don't appear to be perfectly stable, we'll just rely on the presence of the img tokens
    # if input_features:
    #     the_hash.update(str(input_features.tolist()).encode())
    return the_hash.hexdigest()


class CachingAutoModel:
    """
    For HF models, we don't want to load the model unless we need it. Duck-type the attributes we do need.

    Pass as the model_cls for the HF engine and lazy-load the real engine if we need to.
    """

    def __init__(self, model_id, config, generation_config, tokenizer, **model_kwargs):
        # ducktyping
        self.config = config
        self.dtype = self.config.dtype
        self.device = SimpleNamespace(type="cpu")
        self.generation_config = generation_config

        # lazy internals
        self._model_id = model_id
        self._model = None
        self._model_init_kwargs = model_kwargs

        # human helpers
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        # remove some kwargs that from_config doesn't like
        config_kwargs = kwargs.copy()
        for key in ("torch_dtype", "dtype", "quantization_config"):
            config_kwargs.pop(key, None)

        config, config_unused_kwargs = AutoConfig.from_pretrained(model_id, return_unused_kwargs=True, **config_kwargs)
        generation_config, config_unused_kwargs = GenerationConfig.from_pretrained(
            model_id, return_unused_kwargs=True, **(kwargs | config_unused_kwargs)
        )
        # load a copy of the tokenizer here too for human-readable cache
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return cls(
            model_id=model_id,
            config=config,
            generation_config=generation_config,
            tokenizer=tokenizer,
            **(kwargs | config_unused_kwargs),
        )

    # ducktyping methods
    dummy = lambda *_, **__: None

    eval = dummy  # we do this explicitly below
    to = dummy  # we already do device_map=auto by default so we don't need this

    # lazy load the model on generate request
    def _ensure_real_model_loaded(self):
        if self._model is not None:
            return
        print(f"##### LOADING MODEL #####\n{self._model_id}")
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, **self._model_init_kwargs)
        self._model.eval()

    def generate(self, **kwargs):
        # find the cache file for the request
        cache_key = cache_key_for_local_generate(kwargs["input_ids"], kwargs.get("input_features"))
        cache_dir = cache_dir_for_local_generate(self._model_id, cache_key)
        prompt_path = cache_dir / "prompt.txt"
        response_tokens_path = cache_dir / "response.tokens.json"
        response_text_path = cache_dir / "response.content.txt"  # for human readability

        # save the prompt to the cache dir
        input_ids = kwargs["input_ids"]
        streamer = kwargs.get("streamer")
        prompt_text = self._tokenizer.decode(input_ids[0])

        printable_kwargs = kwargs.copy()
        printable_kwargs.pop("streamer", None)  # streamer repr has a pointer which changes per run
        torch.set_printoptions(linewidth=999, profile="full")
        printed_kwargs = pprint.pformat(printable_kwargs, sort_dicts=False, width=120)
        printed_kwargs = re.sub(r",\s+device='.+?'", "", printed_kwargs)  # tensor can be on cpu, mps, or cuda
        prompt_path.write_text(f"{prompt_text}\n==========\n{printed_kwargs}")

        # return the cached resp
        if response_tokens_path.exists():
            with open(response_tokens_path) as f:
                tokens = json.load(f)
            print(f"Returning cached response from: {cache_dir}")
            # if we have streamer, put it 1 token at a time, then return
            if streamer:
                # put the prompt as one big batch first
                prompt_len = len(input_ids[0])
                streamer.put(torch.tensor(tokens[:prompt_len]))
                for token in tokens[prompt_len:]:
                    streamer.put(torch.tensor([token]))
                streamer.end()
            return [tokens]

        # either pass it on or explode
        elif DO_REAL_LOCAL_GENERATE:
            print("Generating new response!")
            self._ensure_real_model_loaded()
            tokens = self._model.generate(**kwargs)
            # save to cache
            with open(response_tokens_path, "w") as f:
                json.dump(tokens[0].tolist(), f)
            response_text = self._tokenizer.decode(tokens[0])  # includes the prompt
            response_text_path.write_text(response_text)
            print(f"New response saved to cache at: {cache_dir}")
            return tokens
        raise ValueError(
            f"Prompt was not cached: {prompt_path}. This may mean the prompt has changed, or you need to hydrate the"
            " prompt cache. Use `KANI_E2E_HYDRATE=local pytest -m e2e` to hydrate local LLM cache."
        )


# --- model capabilities ---
# list of tags: reasoning, function_calling, mm_image, mm_audio, mm_video
# for a test to request a model with a certain tag, use @pytest.mark.request_model_capabilities([tags...])
def _skip_if_missing_capabilities(model_id, model_info, request):
    capabilities = model_info.get("capabilities", [])

    # skip if the model does not have the requested capabilities
    marker = request.node.get_closest_marker("request_model_capabilities")
    if marker:
        requested_capabilities = marker.args[0]
        missing_capabilities = set(requested_capabilities).difference(capabilities)
        if missing_capabilities:
            pytest.skip(f"{model_id} model is missing the following capabilities: {missing_capabilities}")


# ==== define the engine fixtures ====
# --- API ---
# ANTHROPIC
ANTHROPIC_MODELS_TO_TEST = {
    "claude-haiku-4-5": {"capabilities": ["function_calling", "mm_image"]},
}


@pytest.fixture(scope="session", params=list(ANTHROPIC_MODELS_TO_TEST.keys()))
async def _anthropic_engine(request):
    model_id = request.param
    engine = AnthropicEngine(
        model=model_id,
        client=AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"), http_client=httpx.AsyncClient(transport=AsyncCachingTransport())
        ),
    )
    yield engine
    await engine.close()


@pytest.fixture(scope="function")
async def e2e_anthropic_engine(request, _anthropic_engine):
    """
    Parameterized to test multiple different HF engines.

    This is a function-scoped fixture that requests the session-scoped fixture to correctly skip tests based on model
    capabilities.
    """
    model_id = _anthropic_engine.model
    model_info = ANTHROPIC_MODELS_TO_TEST[model_id]
    _skip_if_missing_capabilities(model_id, model_info, request)
    yield _anthropic_engine


# GOOGLE
GOOGLE_MODELS_TO_TEST = {
    "gemini-2.5-flash": {"capabilities": ["function_calling", "mm_image", "mm_audio", "mm_video"]},
}


@pytest.fixture(scope="session", params=list(GOOGLE_MODELS_TO_TEST.keys()))
async def _google_engine(request):
    model_id = request.param
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=genai.types.HttpOptions(async_client_args={"transport": AsyncCachingTransport()}),
    )
    engine = GoogleAIEngine(model=model_id, client=client, multimodal_upload_bytes_threshold=1_000_000_000)
    yield engine
    await engine.close()


@pytest.fixture(scope="function")
async def e2e_google_engine(request, _google_engine):
    model_id = _google_engine.model
    model_info = GOOGLE_MODELS_TO_TEST[model_id]
    _skip_if_missing_capabilities(model_id, model_info, request)
    yield _google_engine


# OPENAI
OPENAI_MODELS_TO_TEST = {
    "gpt-5-mini": {"capabilities": ["function_calling", "mm_image"]},
}


@pytest.fixture(scope="session", params=list(OPENAI_MODELS_TO_TEST.keys()))
async def _openai_engine(request):
    model_id = request.param
    engine = OpenAIEngine(
        model=model_id,
        client=AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), http_client=httpx.AsyncClient(transport=AsyncCachingTransport())
        ),
    )
    yield engine
    await engine.close()


@pytest.fixture(scope="function")
async def e2e_openai_engine(request, _openai_engine):
    model_id = _openai_engine.model
    model_info = OPENAI_MODELS_TO_TEST[model_id]
    _skip_if_missing_capabilities(model_id, model_info, request)
    yield _openai_engine


# --- local ---
class LocalEngineManager:
    """helper to ensure only one local model is loaded on GPU at a time"""

    last_loaded_engine = None
    _lock = asyncio.Lock()

    @classmethod
    async def ensure_closed(cls):
        async with cls._lock:
            if cls.last_loaded_engine is not None:
                await cls.last_loaded_engine.close()
                cls.last_loaded_engine = None


HF_MODELS_TO_TEST = {
    # 2023-2024 chat models
    "meta-llama/Llama-2-7b-chat-hf": {},
    "meta-llama/Llama-3.1-8B-Instruct": {},  # technically can do FC, but it's quite flaky
    "mistralai/Mistral-7B-Instruct-v0.3": {},
    "mistralai/Mistral-Small-Instruct-2409": {"capabilities": ["function_calling"]},
    "google/gemma-3-12b-it": {"kwargs": {"max_context_size": 128000}},  # can do mm_image, but something is borked in HF
    # 2025 thinking models, function calling
    "openai/gpt-oss-20b": {"capabilities": ["reasoning", "function_calling"]},
    # 2025 multimodal models
    # todo
}


# https://docs.pytest.org/en/stable/how-to/fixtures.html#automatic-grouping-of-tests-by-fixture-instances
@pytest.fixture(scope="session", params=list(HF_MODELS_TO_TEST.keys()))
async def _hf_engine(request):
    model_id = request.param
    model_info = HF_MODELS_TO_TEST[model_id]
    load_kwargs = model_info.get("kwargs", {})

    # load the model
    await LocalEngineManager.ensure_closed()
    engine = HuggingEngine(model_id=model_id, model_cls=CachingAutoModel, device=CI_TORCH_DEVICE, **load_kwargs)
    if wrapper := model_specific.parser_for_hf_model(model_id):
        engine = wrapper(engine)
    LocalEngineManager.last_loaded_engine = engine
    yield engine
    await LocalEngineManager.ensure_closed()


@pytest.fixture(scope="function")
async def e2e_huggingface_engine(request, _hf_engine):
    model_id = _hf_engine.model_id
    model_info = HF_MODELS_TO_TEST[model_id]
    _skip_if_missing_capabilities(model_id, model_info, request)
    yield _hf_engine


@pytest.fixture(scope="session", params=["unsloth/gpt-oss-20b-GGUF"])
async def e2e_llamacpp_engine(request):
    """
    llama.cpp doesn't support monkey-patching to return cached responses like we want, so we skip if we aren't
    hydrating real results
    """
    if not DO_REAL_LOCAL_GENERATE:
        pytest.skip("llama.cpp cannot be mocked, set KANI_E2E_HYDRATE=local to run local llamacpp tests")
    model_id = request.param
    await LocalEngineManager.ensure_closed()
    engine = LlamaCppEngine(
        repo_id=model_id,
        filename="*Q4_K_M*",
        model_load_kwargs={"n_gpu_layers": -1},
        prompt_pipeline=model_specific.prompt_pipeline_for_hf_model(model_id),
    )
    if wrapper := model_specific.parser_for_hf_model(model_id):
        engine = wrapper(engine)
    LocalEngineManager.last_loaded_engine = engine
    yield engine
    await LocalEngineManager.ensure_closed()
