kani (カニ)
===========

kani (カニ) is a lightweight and highly hackable harness for chat-based language models with tool usage/function calling.

Compared to other LM harnesses, kani is less opinionated and offers more fine-grained customizability
over the parts of the control flow that matter, making it the perfect choice for NLP researchers, hobbyists, and
developers alike.

.. todo information about the paper and citations

.. toctree::
    :maxdepth: 2
    :caption: Pages

    install (and models)
    kani (entrypoints)
    function_calling (and how to @ai_function)
    customization (overrides and implementing them)
    engines (the builtin ones and how to build them)
    advanced (sub-kanis)
    api_reference (autodoc)

Features
--------

- **Lightweight and high-level** - kani implements common boilerplate to interface with language models without forcing
  you to use opinionated prompt frameworks or complex library-specific tooling.
- **Automatic chat memory management** - Allow chat sessions to flow without worrying about managing the number of
  tokens in the history - kani takes care of it.
- **Function calling with model feedback and retry** - Give models access to functions in just one line of code.
  kani elegantly provides feedback about hallucinated parameters and errors and allows the model to retry calls.
- **Model agnostic** - kani provides a simple interface to implement: token counting and completion generation.
  Implement these two, and kani can run with any language model.
- **You are in control** - You have the ability to override and provide a custom implementation for all
  of these features, allowing you to run experiments just the way you want to.
- **Fast to iterate and intuitive to learn** - With kani, you only write Python - we handle the rest.
- **Asynchronous design from the start** - kani can scale to run multiple chat sessions in parallel easily, without
  having to manage multiple processes or programs.

Quickstart
----------
kani requires Python 3.10 or above.

.. code-block:: console

    $ pip install "kani[openai]"

.. code-block:: python

    import os

    from kani import Kani, chat_in_terminal
    from kani.engines import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    ai = Kani(engine)
    chat_in_terminal(ai)

kani makes the time to set up a working chat model short, while offering the programmer deep customizability over
every prompt, function call, and even the underlying language model.

To learn more about how to customize kani, read on!
