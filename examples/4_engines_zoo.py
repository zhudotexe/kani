# fmt: off
# isort: skip_file
"""
This example showcases all the different engines available and how you can switch between them.
"""

import os

from kani import Kani, chat_in_terminal
# ==== OpenAI (GPT) ====
from kani.engines.openai import OpenAIEngine

engine = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# ==== Anthropic (Claude) ====
# see https://docs.anthropic.com/claude/docs/models-overview for a list of model IDs
from kani.engines.anthropic import AnthropicEngine
engine = AnthropicEngine(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-latest")

# ========== Hugging Face ==========
# ---- Any Model (Chat Templates) ----
from kani.engines.huggingface import HuggingEngine
engine = HuggingEngine(model_id="org-id/model-id")

# ---- DeepSeek R1 (Hugging Face) ----
from kani.engines.huggingface import HuggingEngine
from kani.tool_parsers.deepseek import DeepSeekR1ToolCallParser
# this method is the same for all distills of R1 as well - simply replace the model ID!
model = HuggingEngine(model_id="deepseek-ai/DeepSeek-R1")
engine = DeepSeekR1ToolCallParser(model)

# ---- LLaMA 3 (Hugging Face) ----
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

# ---- Mistral Small/Large (Hugging Face) ----
from kani.engines.huggingface import HuggingEngine
from kani.prompts.impl.mistral import MISTRAL_V3_PIPELINE
from kani.tool_parsers.mistral import MistralToolCallParser
# small (22B):  mistralai/Mistral-Small-Instruct-2409
# large (123B): mistralai/Mistral-Large-Instruct-2407
model = HuggingEngine(model_id="mistralai/Mistral-Small-Instruct-2409", prompt_pipeline=MISTRAL_V3_PIPELINE)
engine = MistralToolCallParser(model)

# ---- Command R (Hugging Face) ----
from kani.engines.huggingface.cohere import CommandREngine
engine = CommandREngine(model_id="CohereForAI/c4ai-command-r-08-2024")

# --------- older models ----------
# ---- LLaMA v2 (Hugging Face) ----
from kani.engines.huggingface.llama2 import LlamaEngine
engine = LlamaEngine(model_id="meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)  # log in with huggingface-cli

# ---- Mistral-7B (Hugging Face) ----
# v0.3 (supports function calling):
from kani.engines.huggingface import HuggingEngine
from kani.prompts.impl.mistral import MISTRAL_V3_PIPELINE
from kani.tool_parsers.mistral import MistralToolCallParser
model = HuggingEngine(model_id="mistralai/Mistral-7B-Instruct-v0.3", prompt_pipeline=MISTRAL_V3_PIPELINE)
engine = MistralToolCallParser(model)

# ========== llama.cpp ==========
# ---- Any Model (Chat Templates) ----
from kani.engines.huggingface import ChatTemplatePromptPipeline
from kani.engines.llamacpp import LlamaCppEngine
pipeline = ChatTemplatePromptPipeline.from_pretrained("org-id/base-model-id")
engine = LlamaCppEngine(repo_id="org-id/quant-model-id", filename="*.your-quant-type.gguf", prompt_pipeline=pipeline)
# NOTE: if the quantized model is sharded in multiple files (e.g. *-00001-of-0000X.gguf), pass the full filename of the
# first shard only.

# ---- DeepSeek R1 (2bit quantized) ----
from kani.engines.huggingface import ChatTemplatePromptPipeline
from kani.engines.llamacpp import LlamaCppEngine
# NOTE: due to an issue in llama-cpp-python you will need to download the files by running the huggingface-cli command
# manually:
# $ huggingface-cli download unsloth/DeepSeek-R1-GGUF --include DeepSeek-R1-Q2_K_XS/*.gguf
pipeline = ChatTemplatePromptPipeline.from_pretrained("deepseek-ai/DeepSeek-R1")
engine = LlamaCppEngine(
    repo_id="unsloth/DeepSeek-R1-GGUF",
    filename="DeepSeek-R1-Q2_K_XS/DeepSeek-R1-Q2_K_XS-00001-of-00005.gguf",
    prompt_pipeline=pipeline,
    model_load_kwargs={"n_gpu_layers": -1, "additional_files": []},
)

# ---- LLaMA v2 (llama.cpp) ----
from kani.engines.llamacpp import LlamaCppEngine
from kani.prompts.impl import LLAMA2_PIPELINE
engine = LlamaCppEngine(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF", filename="*.Q4_K_M.gguf", prompt_pipeline=LLAMA2_PIPELINE
)

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
