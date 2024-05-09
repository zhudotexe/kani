# fmt: off
# isort: skip_file
"""
This example showcases all the different engines available and how you can switch between them.
"""

import os

from kani import Kani, chat_in_terminal

# ==== OpenAI (GPT) ====
from kani.engines.openai import OpenAIEngine
engine = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# ==== Anthropic (Claude) ====
# see https://docs.anthropic.com/claude/docs/models-overview for a list of model IDs
from kani.engines.anthropic import AnthropicEngine
engine = AnthropicEngine(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-opus-20240229")

# ========== Hugging Face ==========
# ---- LLaMA v3 (Hugging Face) ----
import torch
from kani.engines.huggingface import HuggingEngine
from kani.prompts.impl import LLAMA3_PIPELINE
engine = HuggingEngine(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_pipeline=LLAMA3_PIPELINE,
    use_auth_token=True,  # log in with huggingface-cli
    # suggested args from the Llama model card
    model_load_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
)

# NOTE: If you're running transformers<4.40 and LLaMA 3 continues generating after the <|eot_id|> token,
# add `eos_token_id=[128001, 128009]` or upgrade transformers

# ---- LLaMA v2 (Hugging Face) ----
from kani.engines.huggingface.llama2 import LlamaEngine
engine = LlamaEngine(model_id="meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)  # log in with huggingface-cli

# ---- Mixtral-8x22B (Hugging Face) ----
from kani.engines.huggingface import HuggingEngine
from kani.prompts.impl.mistral import MISTRAL_V3_PIPELINE, MixtralFunctionCallingAdapter
model = HuggingEngine(
    model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
    prompt_pipeline=MISTRAL_V3_PIPELINE,
    model_load_kwargs={"device_map": "auto", "torch_dtype": torch.bfloat16},
)

# to enable function calling:
# NOTE: as of May 2024, the huggingface implementation of Mixtral-8x22B function calling is broken:
# https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/discussions/27
# this comment will be removed once it is fixed - until then, you should use another backend
engine = MixtralFunctionCallingAdapter(model)

# ---- Mistral-7B (Hugging Face) ----
from kani.engines.huggingface import HuggingEngine
from kani.prompts.impl import MISTRAL_V1_PIPELINE
engine = HuggingEngine(model_id="mistralai/Mistral-7B-Instruct-v0.2", prompt_pipeline=MISTRAL_V1_PIPELINE)

# Also use the MISTRAL_V1_PIPELINE for Mixtral-8x7B (i.e. mistralai/Mixtral-8x7B-Instruct-v0.1).

# ---- Command R (Hugging Face) ----
import torch
from kani.engines.huggingface.cohere import CommandREngine
engine = CommandREngine(
    model_id="CohereForAI/c4ai-command-r-v01", model_load_kwargs={"device_map": "auto", "torch_dtype": torch.float16}
)

# ---- Gemma (Hugging Face) ----
from kani.engines.huggingface import HuggingEngine
from kani.prompts.impl import GEMMA_PIPELINE
engine = HuggingEngine(model_id="google/gemma-1.1-7b-it", prompt_pipeline=GEMMA_PIPELINE, use_auth_token=True)

# ---- Vicuna v1.3 (Hugging Face) ----
from kani.engines.huggingface.vicuna import VicunaEngine
engine = VicunaEngine(model_id="lmsys/vicuna-7b-v1.3")

# ========== llama.cpp ==========
# ---- LLaMA v2 (llama.cpp) ----
from kani.engines.llamacpp import LlamaCppEngine
engine = LlamaCppEngine(repo_id="TheBloke/Llama-2-7B-Chat-GGUF", filename="*.Q4_K_M.gguf")

# ---- Mistral-7B (llama.cpp) ----
from kani.engines.llamacpp import LlamaCppEngine
from kani.prompts.impl import MISTRAL_V1_PIPELINE
engine = LlamaCppEngine(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", filename="*.Q4_K_M.gguf", prompt_pipeline=MISTRAL_V1_PIPELINE
)

# take your pick - the kani interface is compatible with all!
ai = Kani(engine)

if __name__ == "__main__":
    chat_in_terminal(ai)

# fmt: on
