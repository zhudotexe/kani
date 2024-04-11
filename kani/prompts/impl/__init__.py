"""Implementations of pipelines for popular models."""

from .gemma import GEMMA_PIPELINE
from .llama2 import LLAMA2_PIPELINE
from .vicuna import VICUNA_PIPELINE

MISTRAL_PIPELINE = LLAMA2_PIPELINE
