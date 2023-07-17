Basic Usage
===========
Let's take a look back at the quickstart program:

.. code-block:: python

    from kani import Kani, chat_in_terminal
    from kani.engines.openai import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    ai = Kani(engine)
    chat_in_terminal(ai)

kani is comprised of two main parts: the *engine*, which is the interface between kani and the language model,
and the *kani*, which is responsible for tracking chat history, prompting the engine, and handling function calls.

Kani
----

.. seealso::

    The :class:`.Kani` API documentation.

To initialize a kani, only the ``engine`` is required, though you can configure much more:

.. automethod:: kani.Kani.__init__
    :noindex:

.. code-block:: pycon

    >>> from kani import Kani, chat_in_terminal
    >>> from kani.engines.openai import OpenAIEngine
    >>> api_key = "sk-..."
    >>> engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    >>> ai = Kani(engine, system_prompt="You are a sarcastic assistant.")
    >>> chat_in_terminal(ai, rounds=1)
    USER: Hello kani!
    AI: Is there something I can assist you with today, or are you just here for more of my delightful company?

Entrypoints
^^^^^^^^^^^
While :func:`.chat_in_terminal` is helpful in development, let's look at how to use a :class:`.Kani` in a larger
application.

The two standard entrypoints are :meth:`.Kani.chat_round` and :meth:`.Kani.full_round`, and their ``_str`` counterparts:

.. automethod:: kani.Kani.chat_round
    :noindex:

.. automethod:: kani.Kani.full_round
    :noindex:

.. hint::

    These are asynchronous methods, which means you'll need to be in an async context.

    Web frameworks like FastAPI and Flask 2 allow your route methods to be async, meaning you can await a kani method
    from within your route method without having to get too in the weeds with asyncio.

    Otherwise, you can create an async context by defining an async function and using :func:`asyncio.run`:

    .. code-block:: python

        from kani import Kani, chat_in_terminal
        from kani.engines.openai import OpenAIEngine

        api_key = "sk-..."
        engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
        ai = Kani(engine, system_prompt="You are a helpful assistant.")

        async def chat_with_kani():
            message = await ai.chat_round_str("Hello kani!")
            print(message)
            await ai.close()

        asyncio.run(chat_with_kani())

.. seealso::

    The source code of :func:`.chat_in_terminal`.

Engines
^^^^^^^
This table lists the engines built in to kani:

.. include:: shared/engine_table.rst

.. seealso::

    We won't go too far into implementation details here - if you are interested in implementing your own engine, check
    out :doc:`engines` or the :class:`.BaseEngine` API documentation.

When you are finished with an engine, release its resources with :meth:`.BaseEngine.close`.

Chat Messages
^^^^^^^^^^^^^
Each message contains the ``role`` (a :class:`.ChatRole`: system, assistant, user, or function) that sent the message
and the ``content`` of the message. Optionally, a user message can also contain a ``name`` (for multi-user
conversations), and an assistant message can contain a ``function_call`` (discussed in :doc:`function_calling`).

At a high level, a :class:`.Kani` is responsible for managing a list of :class:`.ChatMessage`: the chat session associated
with it. You can access the chat messages through the :attr:`.Kani.chat_history` attribute.

You may even modify the chat history (i.e. append or delete ChatMessages) to change the prompt at any time.

.. code-block:: pycon

    >>> from kani import Kani, chat_in_terminal
    >>> from kani.engines.openai import OpenAIEngine
    >>> api_key = "sk-..."
    >>> engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    >>> ai = Kani(engine, system_prompt="You are a helpful assistant.")
    >>> chat_in_terminal(ai, rounds=1)
    USER: Hello kani!
    AI: Hello! How can I assist you today?
    >>> ai.chat_history
    [
        ChatMessage(role=ChatRole.USER, content="Hello kani!"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hello! How can I assist you today?"),
    ]
    >>> await ai.get_truncated_chat_history()
    # The system prompt is passed to the engine, but not chat_history - this will be useful later in advanced use cases.
    [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=ChatRole.USER, content="Hello kani!"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hello! How can I assist you today?"),
    ]

Next Steps
----------
In the next section, we'll look at subclassing :class:`.Kani` in order to supply functions to the language model.
Then, we'll look at how you can override and/or extend the implementations of kani methods to control each part of
a chat round.
