"""Various internal utilities to interface with the HF API."""

import functools

from huggingface_hub import RepoCard


@functools.cache
def get_base_models(model_id: str) -> list[str]:
    """Get the base model(s) of a model by reading its model card."""
    model_card = RepoCard.load(model_id)
    base = getattr(model_card.data, "base_model", None)
    if not base:
        return []
    if not isinstance(base, list):
        return [base]
    return base
