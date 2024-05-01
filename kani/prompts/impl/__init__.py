"""Implementations of pipelines for popular models."""

from .gemma import GEMMA_PIPELINE
from .llama2 import LLAMA2_PIPELINE
from .llama3 import LLAMA3_PIPELINE
from .mistral import MISTRAL_V3_PIPELINE
from .vicuna import VICUNA_PIPELINE

MISTRAL_V1_PIPELINE = LLAMA2_PIPELINE
