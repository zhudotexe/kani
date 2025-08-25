"""
Main CLI entrypoint.
"""

import argparse
import importlib
import pkgutil
import sys

from kani import Kani, chat_in_terminal
from kani.utils.cli import create_engine_from_cli_arg

OPTIONAL_LIBS = [
    ("kani.ext.multimodal_core", "__version__"),
    ("anthropic", "__version__"),
    ("google.genai", "__version__"),
    ("llama_cpp", "__version__"),
    ("openai", "VERSION"),
    ("transformers", "__version__"),
]
CLI_EXAMPLES = (
    "examples:\n"
    "  kani openai:gpt-4.1-nano\n"
    "  kani huggingface:meta-llama/Meta-Llama-3-8B-Instruct\n"
    "  kani anthropic:claude-sonnet-4-0\n"
    "  kani google:gemini-2.5-flash"
)


# ==== main ====
def print_logo():
    print(r""" _               _ 
| | ____ _ _ __ (_)
| |/ / _` | '_ \| |
|   | (_| | | | | |
|_|\_\__,_|_| |_|_|""")


def print_version():
    """Entrypoint for kani --version"""
    from ._version import __version__

    # print versions
    # ==== environment ====
    print("{:=^48}".format(" env "))
    print(f"kani {__version__}")
    print(f"Python {sys.version} on {sys.platform}")
    print(f"env: {sys.prefix}")

    # ==== optionals ====
    print("{:=^48}".format(" optionals "))
    max_optional_len = max(len(lib) for lib, _ in OPTIONAL_LIBS)
    for lib, attr in OPTIONAL_LIBS:
        try:
            mod = importlib.import_module(lib)
            version = getattr(mod, attr, "error!")
            print(f"{lib:<{max_optional_len}}: {version}")
        except ImportError:
            print(f"{lib:<{max_optional_len}}: False")

    # --- ext ---
    try:
        import kani.ext

        exts = [name for finder, name, ispkg in pkgutil.iter_modules(kani.ext.__path__, kani.ext.__name__ + ".")]
        if exts:
            print(f"{'kani extensions':<{max_optional_len}}: {'; '.join(exts)}")
    except ImportError:
        pass

    # ==== torch ====
    print("{:=^48}".format(" torch "))
    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False
    print(f"PyTorch: {has_torch}")
    if has_torch:
        torch.utils.collect_env.main()


def chat(arg: str):
    """Standard chat CLI entrypoint"""
    if ":" not in arg:
        # print usage
        print("CLI Usage: kani <provider>:<model_id>\n\n{CLI_EXAMPLES}")
        sys.exit(1)
    try:
        engine = create_engine_from_cli_arg(arg)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(f"Initialized engine: {engine}")
    ai = Kani(engine)
    chat_in_terminal(ai)


def main():
    """The main CLI entrypoint."""
    print_logo()
    parser = argparse.ArgumentParser(
        description=(
            "Use Kani to chat with an LLM in your terminal.\n\nValid engine IDs:\n"
            "* openai (aliases: oai)\n"
            "* anthropic (aliases: ant, claude)\n"
            "* google (aliases: g, gemini)\n"
            "* huggingface (aliases: hf)\n"
        ),
        epilog=CLI_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("provider_and_model", nargs="?", metavar="<provider>:<model_id>")
    parser.add_argument("-V", "--version", action="store_true", help="show version and exit")
    args = parser.parse_args()

    if args.version:
        print_version()
    elif not args.provider_and_model:
        parser.print_help()
    else:
        chat(args.provider_and_model)
