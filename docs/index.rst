kani (カニ)
===========

kani (カニ) is a lightweight and highly hackable framework for chat-based language models with tool usage/function
calling.

Compared to other LM frameworks, kani is less opinionated and offers more fine-grained customizability
over the parts of the control flow that matter, making it the perfect choice for NLP researchers, hobbyists, and
developers alike.

kani comes with support for the following models out of the box, with a **model-agnostic** framework to add support for
many more:

* OpenAI Models (``pip install "kani[openai]"``)
* Anthropic Models (``pip install "kani[anthropic]"``)
* Google AI Models (``pip install "kani[google]"``)
* and _every_ chat model available on Hugging Face through ``transformers`` or ``llama.cpp``!
  (``pip install "kani[huggingface]"``)


Quickstart
----------
kani requires Python 3.10 or above. To install model-specific dependencies, kani uses various extras (brackets after
the library name in ``pip install``). To determine which extra(s) to install, see the
`model table <https://kani.readthedocs.io/en/latest/engines.html>`_, or use the ``[all]`` extra to install everything.

First, install the library. In this quickstart, we'll use the OpenAI engine, though kani is model-agnostic.

.. code-block:: console

    $ pip install "kani[openai]"

Then, let's use kani to create a simple chatbot using ChatGPT as a backend.

.. code-block:: python

    # import the library
    import asyncio
    from kani import Kani, chat_in_terminal
    from kani.engines.openai import OpenAIEngine

    # Replace this with your OpenAI API key: https://platform.openai.com/account/api-keys
    api_key = "sk-..."

    # kani uses an Engine to interact with the language model. You can specify other model
    # parameters here, like temperature=0.7.
    engine = OpenAIEngine(api_key, model="gpt-5-nano")

    # The kani manages the chat state, prompting, and function calling. Here, we only give
    # it the engine to call ChatGPT, but you can specify other parameters like
    # system_prompt="You are..." here.
    ai = Kani(engine)

    # kani comes with a utility to interact with a kani through your terminal...
    chat_in_terminal(ai)

    # or you can use kani programmatically in an async function!
    async def main():
        resp = await ai.chat_round("What is the airspeed velocity of an unladen swallow?")
        print(resp.text)

    asyncio.run(main())

kani makes the time to set up a working chat model short, while offering the programmer deep customizability over
every prompt, function call, and even the underlying language model.

To learn more about how to customize kani with your own prompt wrappers, function calling, and more, read on!

Hands-on examples are available in the `kani repository <https://github.com/zhudotexe/kani/tree/main/examples>`_.

.. toctree::
    :maxdepth: 2
    :caption: Docs

    install
    kani
    function_calling
    customization
    engines
    advanced
    api_reference
    engine_reference
    genindex

.. toctree::
    :maxdepth: 2
    :caption: Community

    community/contributing
    community/extensions
    community/showcase
    Discord <https://discord.gg/eTepTNDxYT>
    Paper <https://aclanthology.org/2023.nlposs-1.8/>
