"""
This test suite:
- lists the 10 most popular text generation models on HF
- downloads their tokenizers
- instantiates a ChatTemplatePromptPipeline for each
- ensures that .explain() works (which calls .execute())
"""

import httpx
import pytest
from transformers import AutoTokenizer

from kani import PromptPipeline
from kani.engines.huggingface.chat_template_pipeline import ChatTemplatePromptPipeline
from kani.prompts.impl import LLAMA3_PIPELINE

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
        raise  # this doesn't actually do anything, it's just here for flow analysis
    pipe = ChatTemplatePromptPipeline(tokenizer)
    pipe.explain()


@pytest.mark.parametrize(
    "pair",
    [
        ("meta-llama/Meta-Llama-3-8B-Instruct", LLAMA3_PIPELINE),
    ],
)
def test_handmade_equivalence(pair: tuple[str, PromptPipeline]):
    model_id, handmade_pipe = pair
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ct_pipe = ChatTemplatePromptPipeline(tokenizer)
    assert ct_pipe(msgs) == handmade_pipe(msgs)
