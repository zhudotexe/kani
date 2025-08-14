"""
Use Kani to chat with an LLM in your terminal.

$ kani <engine-id>:<model-id>

Valid engine IDs:
* openai (aliases: oai)
* anthropic (aliases: ant, claude)
* google (aliases: g, gemini)
* huggingface (aliases: hf)

Examples:
$ kani oai:gpt-4.1-nano
$ kani hf:meta-llama/Meta-Llama-3-8B-Instruct
$ kani ant:claude-sonnet-4-0
$ kani g:gemini-2.5-flash
"""

import kani._cli


if __name__ == "__main__":
    kani._cli.main()
