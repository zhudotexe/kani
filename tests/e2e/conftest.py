"""
E2E engines used in tests.

The engines will use the cache at _cache to return values from a hydrate run, based on the content of the request
to the engine. If a given request is not cached, fail the test.

If the ``KANI_E2E_HYDRATE=api,local`` env var is set, any missing cache entries for one of the E2E mock engines will
make a request to the upstream, and save it for future caching. Remove ``api`` or ``local`` to run a subset.
"""

import hashlib
import io
import json
import os
import pprint
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from kani.engines.anthropic import AnthropicEngine
from kani.engines.google import GoogleAIEngine
from kani.engines.huggingface import HuggingEngine
from kani.engines.openai import OpenAIEngine

# the directory where mock calls & returns are saved; should be committed to Git
MOCK_CACHE_BASE = Path(__file__).parent / "_cache"
DO_REAL_API_REQUESTS = "api" in os.getenv("KANI_E2E_HYDRATE", "")
DO_REAL_LOCAL_GENERATE = "local" in os.getenv("KANI_E2E_HYDRATE", "")


# ==== caching utils ====
# --- http ---
# headers whose values should be saved
KEPT_HEADERS = {"content-type", "accept"}
# headers whose presence should be saved but value redacted
REDACTED_HEADERS = {"authorization", "x-api-key", "x-goog-api-key"}


def cache_dir_for_http_request(url: httpx.URL) -> Path:
    """Get a cache dir per hostname. E.g. _cache/openai.com/"""
    cache_dir = MOCK_CACHE_BASE / url.host
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_key_for_http_request(request: httpx.Request) -> str:
    """Get the cache key (filename) for a given HTTP request."""
    the_hash = hashlib.sha256(fmt_http_request(request).encode())
    *_, last_path_segment = request.url.path.split("/")
    return f"{last_path_segment}-{the_hash.hexdigest()}"


def fmt_http_request(request: httpx.Request) -> str:
    """
    Format an HTTP request in human-readable format.

    Redact any headers in the REDACTED_HEADERS list. Don't save any headers not in the KEPT_HEADERS list.
    """
    # head
    parts = [f"{request.method} {request.url}"]
    for k, v in request.headers.items():
        if k.lower() in REDACTED_HEADERS:
            parts.append(f"{k}: *** REDACTED ***")
        elif k.lower() in KEPT_HEADERS:
            parts.append(f"{k}: {v}")
        else:
            print(f"Discarding header: {k}: {v}")
    # content
    content = request.read()
    if content:
        # hopefully this is text, I think it just explodes if not
        parts.append(f"\n{content.decode()}")
    return "\n".join(parts)


class AsyncCachingTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request):
        # find the cache file for the request
        cache_key = cache_key_for_http_request(request)
        cache_dir = cache_dir_for_http_request(request.url)
        req_path = cache_dir / f"{cache_key}.request.http"
        resp_meta_path = cache_dir / f"{cache_key}.response.meta.json"
        resp_body_path = cache_dir / f"{cache_key}.response.body.bin"

        # save the request to the cache dir
        req_path.write_text(fmt_http_request(request))

        # return the cached resp
        if resp_meta_path.exists() and resp_body_path.exists():
            with open(resp_meta_path) as f:
                meta = json.load(f)
            return httpx.Response(
                status_code=meta["status"], headers=meta["headers"], content=resp_body_path.read_bytes()
            )

        # either pass it on or explode
        if DO_REAL_API_REQUESTS:
            resp = await super().handle_async_request(request)
            # save to cache
            with open(resp_meta_path, "w") as f:
                json.dump({"status": resp.status_code, "headers": resp.headers}, f, indent=2)
            with open(resp_body_path, "wb") as f:
                async for chunk in resp.aiter_bytes(4096):
                    f.write(chunk)
            return resp
        raise ValueError(
            f"Request was not cached: {req_path}. This may mean the request has changed, or you need to hydrate the"
            " request cache. Use `KANI_E2E_HYDRATE=api pytest -m e2e` to hydrate request cache."
        )


# --- tokenwise ---
def cache_dir_for_local_generate(model_id: str) -> Path:
    """Get a cache dir per model name. E.g. _cache/openai__gpt-oss-20b/"""
    cache_dir = MOCK_CACHE_BASE / model_id.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_key_for_local_generate(input_ids: torch.Tensor, input_features: torch.Tensor = None) -> str:
    """Get the cache key (filename) for a given generate request."""
    the_hash = hashlib.sha256()
    input_ids_bytes = io.BytesIO()
    torch.save(input_ids, input_ids_bytes)
    if input_features:
        torch.save(input_features, input_ids_bytes)
    input_ids_bytes.seek(0)
    the_hash.update(input_ids_bytes.getvalue())
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
        self._model_desired_device = None

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

    eval = dummy

    def to(self, device):
        if self._model:
            self._model.to(device)
        self._model_desired_device = device

    # lazy load the model on generate request
    def _ensure_real_model_loaded(self):
        if self._model is not None:
            return
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, **self._model_init_kwargs)
        if self._model_desired_device:
            self._model.to(self._model_desired_device)
        self._model.eval()

    def generate(self, **kwargs):
        # find the cache file for the request
        cache_key = cache_key_for_local_generate(kwargs["input_ids"], kwargs.get("input_features"))
        cache_dir = cache_dir_for_local_generate(self._model_id)
        prompt_path = cache_dir / f"{cache_key}.prompt.txt"
        response_tokens_path = cache_dir / f"{cache_key}.response.tokens.json"
        response_text_path = cache_dir / f"{cache_key}.response.content.txt"  # for human readability

        # save the prompt to the cache dir
        input_ids = kwargs["input_ids"]
        streamer = kwargs.get("streamer")
        prompt_text = self._tokenizer.decode(input_ids[0])
        torch.set_printoptions(linewidth=999, profile="full")
        prompt_path.write_text(f"{prompt_text}\n==========\n{pprint.pformat(kwargs, sort_dicts=False, width=120)}")

        # return the cached resp
        if response_tokens_path.exists():
            with open(response_tokens_path) as f:
                tokens = json.load(f)
            # if we have streamer, put it 1 token at a time, then return
            if streamer:
                for token in tokens:
                    streamer.put(token)
            return [tokens]

        # either pass it on or explode
        elif DO_REAL_LOCAL_GENERATE:
            self._ensure_real_model_loaded()
            tokens = self._model.generate(**kwargs)
            # save to cache
            with open(response_tokens_path, "w") as f:
                json.dump(tokens[0].tolist(), f)
            response_text = self._tokenizer.decode(tokens[0])  # includes the prompt
            response_text_path.write_text(response_text)
            return tokens
        raise ValueError(
            f"Prompt was not cached: {prompt_path}. This may mean the prompt has changed, or you need to hydrate the"
            " prompt cache. Use `KANI_E2E_HYDRATE=local pytest -m e2e` to hydrate local LLM cache."
        )


# ==== define the engine fixtures ====
# --- API ---
@pytest.fixture(scope="session")
async def e2e_anthropic_engine():
    engine = AnthropicEngine(
        model="claude-haiku-4-5",
        client=AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"), http_client=httpx.AsyncClient(transport=AsyncCachingTransport())
        ),
    )
    yield engine
    await engine.close()


@pytest.fixture(scope="session")
async def e2e_google_engine():
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=genai.types.HttpOptions(async_client_args={"transport": AsyncCachingTransport()}),
    )
    engine = GoogleAIEngine(model="gemini-2.5-flash", client=client)
    yield engine
    await engine.close()


@pytest.fixture(scope="session")
async def e2e_openai_engine():
    engine = OpenAIEngine(
        model="gpt-5-mini",
        client=AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), http_client=httpx.AsyncClient(transport=AsyncCachingTransport())
        ),
    )
    yield engine
    await engine.close()


# --- local ---
@pytest.fixture(scope="session")
async def e2e_huggingface_engine_factory():
    """A sync function (model_id) -> HuggingEngine"""
    engine_cache = {}

    def f(model_id):
        if model_id in engine_cache:
            return engine_cache[model_id]
        the_engine = HuggingEngine(model_id=model_id, model_cls=CachingAutoModel)
        engine_cache[model_id] = the_engine
        return the_engine

    yield f
    for _, engine in engine_cache.items():
        await engine.close()


@pytest.fixture(scope="session")
async def e2e_huggingface_engine(e2e_huggingface_engine_factory):
    # yield e2e_huggingface_engine_factory("openai/gpt-oss-20b")
    yield e2e_huggingface_engine_factory("google/gemma-3-1b-it")


@pytest.fixture(scope="session")
async def e2e_llamacpp_engine():
    pass
