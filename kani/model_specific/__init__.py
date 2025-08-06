import fnmatch
import importlib
import logging

from kani.prompts import PromptPipeline
from kani.utils.huggingface import get_base_models

# list of (HF model id glob, path to import)
# the path to import can be either a PromptPipeline instance or a function (tokenizer) => pipeline
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


def prompt_pipeline_for_hf_model(model_id: str, tokenizer=None, search_parents=True, fallback_to_chat_template=True):
    """
    Find and return the correct prompt pipeline for the given HF model:
    If a handwritten pipeline is available, use that one.
    Otherwise, if *search_parents* is True and the model is a fine-tune/quantization of a parent model, recurse on
    the parent model.
    Otherwise, use the model's chat template with default behaviour.
    """
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
            return pipe(tokenizer)

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
        return ChatTemplatePromptPipeline(tokenizer)
    return ChatTemplatePromptPipeline.from_pretrained(model_id)


def maybe_parser_for_hf_model(model_id: str, search_parents=True):
    """
    Find and print a warning about using the correct parser for the given HF model, if one exists and it is not in the
    stack; otherwise return None.
    This is just used for notifying the user that they might want to use the parser instead.
    """
