Basic Usage
===========
Let's take a look back at the quickstart program:

.. code-block:: python

    from kani import Kani, chat_in_terminal
    from kani.engines import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    ai = Kani(engine)
    chat_in_terminal(ai)

kani is comprised of two main parts: the *engine*, which is the interface between kani and the language model,
and the *kani*, which is responsible for tracking chat history, prompting the engine, and handling function calls.

ChatMessage
-----------
Before we dive into what methods are available in the engine and kani, let's take a look at :cls:`ChatMessage`s,
a core component of representing the chat context.

Each message contains the ``role`` (a :cls:`ChatRole`: system, assistant, user, or function) that sent the message
and the ``content`` of the message. Optionally, a user message can also contain a ``name`` (for multi-user
conversations), and an assistant message can contain a ``function_call`` (discussed in :doc:`function_calling`).

Engine
------
This table lists the engines built in to kani:

.. todo: engine template here

.. seealso::

    We won't go too far into implementation details here - if you are interested in implementing your own engine, check
    out :doc:`engines` or the :cls:`BaseEngine` API documentation.

Each engine must implement two methods: :meth:`BaseEngine.message_len`, which takes a single :cls:`ChatMessage` and
returns the length of that message, in tokens, and :meth:`BaseEngine.predict`, which is responsible for taking
a list of :cls:`ChatMessage`s and :cls:`AIFunction`s (discussed in the next section) and returning a new
:cls:`Completion <BaseCompletion>`.

These methods are lower-level and used by :cls:`Kani` to manage the chat, but you can also call them yourself.

Kani
----

.. seealso::

    The :cls:`Kani` API documentation.

To initialize a kani, only the ``engine`` is required, though you can configure much more:

.. todo autodoc Kani.__init__ here, also actually do the examples

.. code-block:: pythonconsole

    >>> from kani import Kani, chat_in_terminal
    >>> from kani.engines import OpenAIEngine
    >>> api_key = "sk-..."
    >>> engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    >>> ai = Kani(engine, system_prompt="You are a sarcastic assistant.")
    >>> chat_in_terminal(ai, rounds=1)
    USER: Hello kani!
    AI: What do you want now?

When you are finished with a kani instance, release its resources with :meth:`Kani.close`.

Programmatic Chat
^^^^^^^^^^^^^^^^^
While :func:`chat_in_terminal` is a helpful toy, let's look at how to use a :cls:`Kani` in a larger application.

The two standard entrypoints are :meth:`Kani.chat_round` and :meth:`Kani.full_round`, and their ``_str`` counterparts:

.. todo autodoc those here

.. hint::

    These are asynchronous methods, which means you'll need to be in an async context.

    Web frameworks like FastAPI and Flask 2 allow your route methods to be async, meaning you can await a kani method
    from within your route method without having to get too in the weeds with asyncio.

    Otherwise, you can create an async context by defining an async function and using :func:`asyncio.run`: .. todo intersphinx

    .. code-block:: python

        from kani import Kani, chat_in_terminal
        from kani.engines import OpenAIEngine

        api_key = "sk-..."
        engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
        ai = Kani(engine, system_prompt="You are a helpful assistant.")

        async def chat_with_kani():
            message = await ai.chat_round("Hello kani!")
            print(message)

        asyncio.run(chat_with_kani())

.. seealso::

    The source code of :func:`chat_in_terminal`.

Chat Messages
^^^^^^^^^^^^^
At a high level, a :cls:`Kani` is responsible for managing a list of :cls:`ChatMessage`s: the chat session associated
with it. You can access the chat messages through the :attr:`Kani.chat_history` attribute.

You may even modify the chat history (i.e. append or delete ChatMessages) to change the prompt at any time.

.. code-block:: pythonconsole

    >>> from kani import Kani, chat_in_terminal
    >>> from kani.engines import OpenAIEngine
    >>> api_key = "sk-..."
    >>> engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    >>> ai = Kani(engine, system_prompt="You are a helpful assistant.")
    >>> chat_in_terminal(ai, rounds=1)
    USER: Hello kani!
    AI: Hello! How can I help?
    >>> ai.chat_history
    [
        ChatMessage(role=ChatRole.USER, content="Hello kani!"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hello! How can I help?"),
    ]
    >>> await ai.get_truncated_chat_history()
    # The system prompt is passed to the engine, but not chat_history - this will be useful later in advanced use cases.
    [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=ChatRole.USER, content="Hello kani!"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hello! How can I help?"),
    ]

Next Steps
----------
In the next section, we'll look at subclassing :cls:`Kani` in order to supply functions to the language model.
Then, we'll look at how you can override and/or extend the implementations of kani methods to control each part of
a chat round.
