import fnmatch
import functools
import importlib
import logging
import sys
import warnings

from .base import BaseToolCallParser

# list of (HF model id glob, path to import)
# the path to import can be either a PromptPipeline instance or a function (tokenizer, **kwargs) => pipeline
PROMPT_PIPELINE_REGISTRY = [
    ("openai/gpt-oss-*", "kani.model_specific.gpt_oss.build_gptoss_prompt_pipeline"),
    ("meta-llama/Llama-2*", "kani.model_specific.llama2.LLAMA2_PIPELINE"),
    ("meta-llama/Llama-3-*", "kani.model_specific.llama3.LLAMA3_PIPELINE"),
    ("mistralai/Mistral-7B*", "kani.model_specific.mistral.MISTRAL_V3_PIPELINE"),
    ("mistralai/Mixtral-8x*", "kani.model_specific.mistral.MISTRAL_V3_PIPELINE"),
]

# list of (HF model id glob, path to import)
PARSER_REGISTRY = [
    ("deepseek-ai/DeepSeek-R1*", "kani.model_specific.deepseek.DeepSeekR1ToolCallParser"),
    ("openai/gpt-oss-*", "kani.model_specific.gpt_oss.GPTOSSParser"),
    ("mistralai/*-2404", "kani.model_specific.mistral.MistralToolCallParser"),
    ("mistralai/*-2405", "kani.model_specific.mistral.MistralToolCallParser"),
    ("mistralai/*-2407", "kani.model_specific.mistral.MistralToolCallParser"),
    ("mistralai/*-2409", "kani.model_specific.mistral.MistralToolCallParser"),
]

log = logging.getLogger(__name__)
# this is a bit hacky, but useful to prevent developer foot-guns
# a global that tracks whether the developer has ever initialize a BaseToolCallParser
# so we can warn them if their model has one that they are not using
_has_initialized_model_specific_parser = False


def prompt_pipeline_for_hf_model(
    model_id: str, tokenizer=None, search_parents=True, fallback_to_chat_template=True, *, chat_template_kwargs=None
):
    """
    Find and return the correct prompt pipeline for the given HF model:
    If a handwritten pipeline is available, use that one.
    Otherwise, if *search_parents* is True and the model is a fine-tune/quantization of a parent model, recurse on
    the parent model.
    Otherwise, use the model's chat template with default behaviour.
    """
    from kani.prompts import PromptPipeline
    from kani.utils.huggingface import get_base_models

    if chat_template_kwargs is None:
        chat_template_kwargs = {}

    for pattern, import_path in PROMPT_PIPELINE_REGISTRY:
        if fnmatch.fnmatch(model_id, pattern):
            # import the thing and return it
            # we have a test to make sure each of these is safely importable
            mod_name, attr_name = import_path.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            pipe = getattr(mod, attr_name)

            log.info(f"A handwritten prompt pipeline was found for the {model_id} model (matching {pattern}).")
            if isinstance(pipe, PromptPipeline):
                return pipe
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_id)
            return pipe(tokenizer, **chat_template_kwargs)

    # if there's a parent and the model is a fine-tune/quantization recurse
    if search_parents:
        parent_model_ids = get_base_models(model_id)
        if len(parent_model_ids) == 1:
            parent_model_id = parent_model_ids[0]
            log.info(f"Searching for prompt pipelines for parent model of {model_id}: {parent_model_id}")
            pipe = prompt_pipeline_for_hf_model(parent_model_id, fallback_to_chat_template=False)
            if pipe:
                return pipe
        else:
            log.info(f"Could not find parent model of {model_id} (got {parent_model_ids})")

    # otherwise use the chat template
    if not fallback_to_chat_template:
        return None
    from kani.engines.huggingface import ChatTemplatePromptPipeline

    log.warning(
        f"No handwritten prompt pipeline was found for the {model_id} model. Kani is falling back to the Hugging Face"
        " chat template. For most models this is okay, but you may want to verify that the chat template correctly"
        " passes tool calls to the LLM."
    )

    if tokenizer:
        return ChatTemplatePromptPipeline(tokenizer, **chat_template_kwargs)
    return ChatTemplatePromptPipeline.from_pretrained(model_id, **chat_template_kwargs)


@functools.cache
def parser_for_hf_model(model_id: str, search_parents=True):
    """
    Find and print a warning about using the correct parser for the given HF model, if one exists and it is not in the
    stack; otherwise return None.
    This is just used for notifying the user that they might want to use the parser instead.
    """
    from kani.utils.huggingface import get_base_models

    for pattern, import_path in PARSER_REGISTRY:
        if fnmatch.fnmatch(model_id, pattern):
            # import the thing and return it
            # we have a test to make sure each of these is safely importable
            mod_name, attr_name = import_path.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            parser = getattr(mod, attr_name)

            log.info(f"A handwritten parser was found for the {model_id} model (matching {pattern}).")
            return parser

    # if there's a parent and the model is a fine-tune/quantization recurse
    if search_parents:
        parent_model_ids = get_base_models(model_id)
        if len(parent_model_ids) == 1:
            parent_model_id = parent_model_ids[0]
            log.info(f"Searching for parsers for parent model of {model_id}: {parent_model_id}")
            parser = parser_for_hf_model(parent_model_id)
            if parser:
                return parser
        else:
            log.info(f"Could not find parent model of {model_id} (got {parent_model_ids})")

    # otherwise return None
    return None


def warn_for_uninitialized_parser(model_id: str):
    """Log a warning if no model-specific parser is initialized and there is a handwritten parser available."""
    if (not _has_initialized_model_specific_parser) and (parser := parser_for_hf_model(model_id)):
        if sys.version_info >= (3, 12):
            _warnings_kwargs = {"skip_file_prefixes": ("kani",)}
        else:
            _warnings_kwargs = {"stacklevel": 2}

        warnings.warn(
            "You are using a model that requires additional parsing of its outputs but no model-specific parser is"
            f" wrapping it. Consider wrapping your engine with {parser!s} in order"
            " to correctly parse tool calls and/or reasoning chunks:\n"
            f">>> from {parser.__module__} import {parser.__name__}\n"
            f">>> engine = {parser.__name__}(<previous engine def>)",
            **_warnings_kwargs,
        )
