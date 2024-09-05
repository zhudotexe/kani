"""
This submodule contains utilities to transform a list of Kani :class:`.ChatMessage` into low-level formats to be
consumed by an engine (e.g. ``str``, ``list[dict]``, or ``torch.Tensor``).
"""

from .base import PipelineStep
from .pipeline import PromptPipeline
from .types import ApplyContext
