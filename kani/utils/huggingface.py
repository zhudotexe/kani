"""Various internal utilities to interface with the HF API."""

import functools
import logging

from huggingface_hub import RepoCard

log = logging.getLogger(__name__)


@functools.cache
def get_base_models(model_id: str) -> list[str]:
    """Get the base model(s) of a model by reading its model card."""
    try:
        model_card = RepoCard.load(model_id)
    except Exception as e:
        log.warning("Could not load model card for listed parent model:", exc_info=e)
        return []
    base = getattr(model_card.data, "base_model", None)
    if not base:
        return []
    if not isinstance(base, list):
        return [base]
    return base
