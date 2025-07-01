"""
Use Kani to chat with an LLM in your terminal.

python -m kani engine-id/model-id

Valid engine IDs:
* openai (aliases: oai)
* anthropic (aliases: ant, claude)
* huggingface (aliases: hf)

Examples:
python -m kani oai/gpt-4.1-nano
python -m kani hf/meta-llama/Meta-Llama-3-8B-Instruct
python -m kani ant/claude-sonnet-4-0
"""

import sys

from kani import Kani, chat_in_terminal


# ==== engine defs ====
def chat_openai(model_id: str):
    from kani.engines.openai import OpenAIEngine

    return OpenAIEngine(model=model_id)


def chat_anthropic(model_id: str):
    from kani.engines.anthropic import AnthropicEngine

    return AnthropicEngine(model=model_id)


def chat_huggingface(model_id: str):
    from kani.engines.huggingface import HuggingEngine

    return HuggingEngine(model_id=model_id)


PROVIDER_MAP = {
    # openai
    "openai": chat_openai,
    "oai": chat_openai,
    # anthropic
    "anthropic": chat_anthropic,
    "ant": chat_anthropic,
    "claude": chat_anthropic,
    # huggingface
    "huggingface": chat_huggingface,
    "hf": chat_huggingface,
}


# ==== main ====
def print_version():
    from kani import _optional
    from ._version import __version__

    # print versions
    print("========== version ==========")
    print(f"kani v{__version__}")
    print(f"Python {sys.version} on {sys.platform}")

    print("========== optionals ==========")
    print(
        "kani-multimodal-core:"
        f" {'False' if not _optional.has_multimodal_core else _optional.multimodal_core.__version__}"
    )

    print("========== torch ==========")
    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False
    print(f"PyTorch: {has_torch}")
    if has_torch:
        torch.utils.collect_env.main()


def chat(arg: str):
    if "/" not in arg:
        # print usage
        print(
            "CLI Usage: python -m kani <provider>/<model_id>\n\n"
            "Examples:\n"
            "python -m kani openai/gpt-4.1-nano\n"
            "python -m kani huggingface/meta-llama/Meta-Llama-3-8B-Instruct\n"
            "python -m kani anthropic/claude-sonnet-4-0"
        )
        sys.exit(1)

    provider, model_id = arg.split("/", 1)
    if provider not in PROVIDER_MAP:
        print(f"Invalid model provider: {provider!r}. Valid options: {list(PROVIDER_MAP)}")
        sys.exit(1)

    engine = PROVIDER_MAP[provider](model_id)
    ai = Kani(engine)
    chat_in_terminal(ai)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_version()
    else:
        chat(sys.argv[1])
