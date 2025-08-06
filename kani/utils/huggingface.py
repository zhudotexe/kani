"""Various internal utilities to interface with the HF API."""

from huggingface_hub import RepoCard


def get_base_models(model_id: str) -> list[str]:
    """Get the base model(s) of a model by reading its model card."""
    model_card = RepoCard.load(model_id)
    base = model_card.data.base_model
    if not base:
        return []
    if not isinstance(base, list):
        return [base]
    return base
