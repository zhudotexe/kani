"""
This test suite:
- lists the 10 most popular text generation models on HF
- downloads their tokenizers
- instantiates a ChatTemplatePromptPipeline for each
- ensures that .explain() works (which calls .execute())
"""

import httpx
import pytest
import torch
from transformers import AutoTokenizer

from kani import ChatMessage, PromptPipeline
from kani.engines.huggingface.chat_template_pipeline import ChatTemplatePromptPipeline
from kani.prompts.impl import GEMMA_PIPELINE, LLAMA2_PIPELINE, LLAMA3_PIPELINE, MISTRAL_V3_PIPELINE

# model IDs to always test the chat templates of
forced_model_ids = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


# get 10 most popular textgen models and test their chat templates
def popular_model_ids():
    http = httpx.Client()
    resp = http.get("https://huggingface.co/api/trending")
    data = resp.json()

    trending_model_ids = [
        m["repoData"]["id"]
        for m in data["recentlyTrending"]
        if m["repoType"] == "model" and m["repoData"].get("pipeline_tag") == "text-generation"
    ]
    return trending_model_ids


@pytest.mark.parametrize("chat_template_model_id", [*forced_model_ids, *popular_model_ids()])
def test_chat_templates(chat_template_model_id: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(chat_template_model_id)
    except ValueError:
        pytest.skip("This model requires untrusted code, skipping")
    except EnvironmentError:
        pytest.skip("This model is gated, skipping")
    if tokenizer.chat_template is None:
        pytest.skip("This model does not have a chat template, skipping")
    pipe = ChatTemplatePromptPipeline(tokenizer)
    pipe.explain()


@pytest.mark.parametrize(
    "model_id,handmade_pipe",
    [
        ("meta-llama/Meta-Llama-3-8B-Instruct", LLAMA3_PIPELINE),
        ("mistralai/Mistral-7B-Instruct-v0.3", MISTRAL_V3_PIPELINE),
        ("meta-llama/Llama-2-7b-chat-hf", LLAMA2_PIPELINE),
        ("google/gemma-1.1-7b-it", GEMMA_PIPELINE),
        # ("lmsys/vicuna-7b-v1.3", VICUNA_PIPELINE),  # no chat template
    ],
)
def test_handmade_equivalence(model_id: str, handmade_pipe: PromptPipeline):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ct_pipe = ChatTemplatePromptPipeline(tokenizer)
    msgs = [  # no system message since not all models support it
        ChatMessage.user("Hello world!"),
        ChatMessage.assistant("I am a robot"),
        ChatMessage.user("What does kani mean in English?"),
    ]
    # run through the pipes
    ct_out = ct_pipe(msgs)
    handmade_out = handmade_pipe(msgs)
    # make sure they are strs
    if isinstance(ct_out, torch.Tensor):
        ct_out = tokenizer.decode(ct_out)
    # assert equivalence
    assert ct_out == handmade_out
