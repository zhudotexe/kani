Changelog
=========

## v1.7.0 - Token Counting Refactor

Under the hood, kani now uses full prompts (i.e., a list of messages + functions) to count tokens, rather than summing
the token counts of messages individually. This makes token counting more reliable for models which do not expose their
tokenizer (e.g. Claude and Gemini) and models with strict chat templates (HF transformers, llama.cpp).

If you do not manually count tokens by using `Kani.message_token_len`, `BaseEngine.message_len`,
`BaseEngine.token_reserve` or `BaseEngine.function_token_reserve`, no change is needed.

If you have implemented your own engine, the methods above are now deprecated. To implement token counting
functionality, replace `BaseEngine.message_len`, `.token_reserve`, and `.function_token_reserve` with the new method
`.prompt_len(messages, functions)`. This method may be asynchronous.

This change aims to make implementing new engines simpler and more streamlined, as prompt-wise token counting can reuse
much of the same code as inference.

### Breaking Changes

- Deprecated `Kani.message_token_len` - use `await Kani.prompt_token_len` instead
- Deprecated `BaseEngine.message_len`, `.token_reserve`, `.function_token_reserve` - implement `BaseEngine.prompt_len`
  instead
- `AIFunction.auto_truncate` now truncates to a certain number of **characters** instead of tokens

### New Features

- Added `BaseEngine.prompt_len` and `Kani.prompt_token_len`
- Added native multimodal support to the `HuggingEngine`
- Added `TextPart` message part for rich extras
- Added documentation for low-level hackability overrides for certain API-based engines (e.g., for server-side tool
  calling)

### Fixes

- Fixed a case where decoding params specified in multiple places could overlap and cause errors
- Fixed an issue where a Kani instance's property getters would be called on construction while searching for
  AIFunctions
- Fixed the process hanging forever if the call to PreTrainedModel.generate crashes during a call to
  HuggingEngine.stream
- Fixed an issue where the LlamaCppEngine would not release its resources immediately when closed
- Fixed an issue when parsing Mistral-style function calls for functions without arguments
- Fixed an issue where certain base models would not be found when trying to automatically identify base models for
  quantized variants
- Removed some unused exception types after the HTTPEngine was removed
- GoogleAIEngine: Raise better warnings when the Gemini API returns an unexpectedly empty response

---

## v1.6.1

- Removed deprecated HTTPClient (since v1.0.0)
- Increased default `desired_response_tokens` to 10% of the model's max context length or 8192 tokens, whichever is
  shorter
- Anthropic: reasoning is now returned in a ReasoningPart instead of an AnthropicUnknownPart
- Anthropic: increased default `max_tokens` to 2048
- Anthropic: fixed tool results not being sent to the model correctly
- OpenAI: better warning about which tokenizer is used when a model ID is not found
- Google AI: reasoning is now returned in a ReasoningPart instead of a str and is forwarded correctly for multi-turn
  function calls

---

## v1.6.0 - Multimodal Inputs, Gemini Support, kani CLI

kani-multimodal-core should be installed alongside the core kani install using an extra:

```shell
$ pip install "kani[multimodal]"
```

However, you can also explicitly specify a version and install the core package itself:

```shell
$ pip install kani-multimodal-core
```

### Features

This package provides the core multimodal extensions that engine implementations can use -- it does not provide any
engine implementations on its own.

The package adds support for:

- Images (`kani.ext.multimodal_core.ImagePart`)
- Audio (`kani.ext.multimodal_core.AudioPart`)
- Video (`kani.ext.multimodal_core.VideoPart`)
- Other binary files, such as PDFs (`kani.ext.multimodal_core.BinaryFilePart`)

When installed, these core kani engines will automatically use the multimodal parts:

- OpenAIEngine
- AnthropicEngine
- GoogleAIEngine

Additionally, the core kani `chat_in_terminal` method will support attaching multimodal data from a local drive or
from the internet using `@/path/to/media` or `@https://example.com/media`.

#### Message Parts

The main feature you need to be familiar with is the `MessagePart`, the core way of sending messages to the engine.
To do this, when you call the kani round methods (i.e. `Kani.chat_round` or `Kani.full_round` or their str variants),
pass a *list* of multimodal parts rather than a string:

```python
from kani import Kani
from kani.engines.openai import OpenAIEngine
from kani.ext.multimodal_core import ImagePart

engine = OpenAIEngine(model="gpt-4.1-nano")
ai = Kani(engine)

# notice how the arg is a list of parts rather than a single str!
msg = await ai.chat_round_str([
    "Please describe this image:",
    ImagePart.from_file("path/to/image.png")
])
print(msg)
```

See the docs (https://kani-multimodal-core.readthedocs.io) for more information about the provided message parts.

#### Terminal Utility

When installed, kani-multimodal-core augments the `chat_in_terminal` utility provided by kani.

This utility allows you to provide multimodal media on your disk or on the internet inline by prepending it with an
@ symbol:

```pycon
>>> from kani import chat_in_terminal
>>> chat_in_terminal(ai)
USER: Please describe this image: @path/to/image.png and also this one: @https://example.com/image.png
```

- Added native support for multimodal (image, video, audio) models using the `kani-multimodal-core`
  package (https://github.com/zhudotexe/kani-multimodal-core)!
    - The AnthropicEngine, OpenAIEngine, and GoogleAIEngine will automatically support multimodal inputs when
      `kani-multimodal-core` is installed

### New feature: Native Google Gemini support

```shell
$ pip install "kani[google]"
```

```python
from kani import Kani
from kani.engines.google import GoogleAIEngine

engine = GoogleAIEngine(model="gemini-2.5-flash")
```

This engine supports all Google AI models through the Google AI Studio API.

See https://ai.google.dev/gemini-api/docs/models for a list of available models.

**Multimodal support**: images, audio, video.

- Added the `GoogleAIEngine` for Google Gemini support - supports function calling & multimodal inputs

### New feature: `kani` CLI tool

When kani is installed, you can run `$ kani provider:model-id` to begin chatting with a model in your terminal!

Examples:

```shell
$ kani openai:gpt-4.1-nano
$ kani huggingface:meta-llama/Meta-Llama-3-8B-Instruct
$ kani anthropic:claude-sonnet-4-0
$ kani google:gemini-2.5-flash
```

This CLI helper automatically creates a Engine and Kani instance, and calls `chat_in_terminal()` so you can test LLMs
faster. Just as with `chat_in_terminal()`, you can use `@/path/to/file` or `@https://example.com/file` to attach
multimodal parts to your CLI inputs.

- Added a `kani` CLI tool for easy chatting in terminal
    - Use `@/path/to/file` or `@https://example.com/file` to upload a multimodal file to your CLI

### Additional features & fixes

- Added `Message.extras` to store arbitrary additional information with a Message object
    - Certain engines will set an extra to store the raw response returned by the API (see engine docs)
    - For example, to access the detailed usage object returned by an `OpenAIEngine`, you can use:

```python
msg = await ai.chat_round(...)  # or .full_round
openai_usage = msg.extra["openai_usage"]
```

- Added a `save_format` parameter to `Kani.save()` to allow saving to a `.kani` file instead of `.json`
    - Saving to a `.kani` file is used by default unless the filename given to `Kani.save()` ends with `.json`
    - A `.kani` file is a ZIP file containing the saved chat state of the Kani instance
    - Certain extensions (e.g., `kani-multimodal-core`) may save additional files to the `.kani` archive to save
      multimodal MessageParts without inflating the size of the saved JSON file
- Fixed the kani CLI not always quitting on ^C

#### Engine-specific

- Anthropic: Handle PDF inputs using `kani.ext.multimodal_core.BinaryFilePart`
- Hugging Face: Load on MPS by default when detected on a macOS system

---

## Releases prior to v1.6.0

For the release notes of versions prior to v1.6.0, see
the [release notes on GitHub](https://github.com/zhudotexe/kani/releases).