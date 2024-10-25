Basic Usage
===========
Let's take a look back at the quickstart program:

.. code-block:: python

    from kani import Kani, chat_in_terminal
    from kani.engines.openai import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-4o-mini")
    ai = Kani(engine)
    chat_in_terminal(ai)

kani is comprised of two main parts: the *engine*, which is the interface between kani and the language model,
and the *kani*, which is responsible for tracking chat history, prompting the engine, and handling function calls.

.. image:: _static/concepts-figure.png
    :width: 60%
    :align: center

In this section, we'll look at how to initialize a Kani class and core concepts in the library.

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
    >>> engine = OpenAIEngine(api_key, model="gpt-4o-mini")
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

.. important::
    These are asynchronous methods, which means you'll need to be in an async context.

Web frameworks like FastAPI and Flask 2 allow your route methods to be async, meaning you can await a kani method
from within your route method without having to get too in the weeds with asyncio.

Otherwise, you can create an async context by defining an async function and using :func:`asyncio.run`. For example,
here's how you might implement a simple chat:

.. code-block:: python

    import asyncio
    from kani import Kani
    from kani.engines.openai import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-4o-mini")
    ai = Kani(engine, system_prompt="You are a helpful assistant.")

    # define your function normally, using `async def` instead of `def`
    async def chat_with_kani():
        while True:
            user_message = input("USER: ")
            # now, you can use `await` to call kani's async methods
            message = await ai.chat_round_str(user_message)
            print("AI:", message)

    # use `asyncio.run` to call your async function to start the program
    asyncio.run(chat_with_kani())

Engines
^^^^^^^
Engines are responsible for interfacing with a language model.

This table lists the engines built in to kani:

.. include:: shared/engine_table.rst

.. seealso::

    We won't go too far into implementation details here - if you are interested in implementing your own engine, check
    out :doc:`engines` or the :class:`.BaseEngine` API documentation.

When you are finished with an engine, release its resources with :meth:`.BaseEngine.close`.

Concept: Chat Messages
----------------------
Each message contains the ``role`` (a :class:`.ChatRole`: system, assistant, user, or function) that sent the message
and the ``content`` of the message. Optionally, a user message can also contain a ``name`` (for multi-user
conversations), and an assistant message can contain a ``function_call`` (discussed in :doc:`function_calling`).

.. autoclass:: kani.ChatMessage
    :members:
    :exclude-members: model_config, model_fields, model_computed_fields
    :class-doc-from: class
    :noindex:

At a high level, a :class:`.Kani` is responsible for managing a list of :class:`.ChatMessage`: the chat session
associated with it. You can access the chat messages through the :attr:`.Kani.chat_history` attribute.

You may even modify the chat history (e.g. append or delete ChatMessages or edit a message's content) to change the
prompt at any time.

.. warning::
    In some advanced use cases, :attr:`.ChatMessage.content` may be a tuple of :class:`.MessagePart` or ``str`` rather
    than a string. ChatMessage exposes :attr:`.ChatMessage.text` (always a string or None) and
    :attr:`.ChatMessage.parts` (always a list of message parts), which we recommend using instead of
    :attr:`.ChatMessage.content`. See :doc:`advanced/messageparts` for more information.

    These properties are dynamically generated based on the underlying content, and it is safe to mix messages
    with different content types in a single Kani.

.. code-block:: pycon

    >>> from kani import Kani, chat_in_terminal
    >>> from kani.engines.openai import OpenAIEngine
    >>> api_key = "sk-..."
    >>> engine = OpenAIEngine(api_key, model="gpt-4o-mini")
    >>> ai = Kani(engine, system_prompt="You are a helpful assistant.")
    >>> chat_in_terminal(ai, rounds=1)
    USER: Hello kani!
    AI: Hello! How can I assist you today?
    >>> ai.chat_history
    [
        ChatMessage(role=ChatRole.USER, content="Hello kani!"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hello! How can I assist you today?"),
    ]
    >>> await ai.get_prompt()
    # The system prompt is passed to the engine, but isn't part of chat_history
    # - this will be useful later in advanced use cases.
    [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=ChatRole.USER, content="Hello kani!"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hello! How can I assist you today?"),
    ]

Streaming
---------
kani supports streaming to print tokens from the engine as they are received. Streaming is designed to be a drop-in
superset of the ``chat_round`` and ``full_round`` methods, allowing you to gradually refactor your code without ever
leaving it in a broken state.

To request a stream from the engine, use :meth:`.Kani.chat_round_stream` or :meth:`.Kani.full_round_stream`. These
methods will return a :class:`.StreamManager`, which you can use in different ways to consume the stream.

The simplest way to consume the stream is to iterate over it with ``async for``, which will yield a stream of
:class:`str`.

.. code-block:: python

    # CHAT ROUND:
    stream = ai.chat_round_stream("What is the airspeed velocity of an unladen swallow?")
    async for token in stream:
        print(token, end="")
    msg = await stream.message()

    # FULL ROUND:
    async for stream in ai.full_round_stream("What is the airspeed velocity of an unladen swallow?"):
        async for token in stream:
            print(token, end="")
        msg = await stream.message()

kani also provides a helper to print streams (:func:`kani.print_stream`):

.. code-block:: python

    stream = ai.chat_round_stream("What is the most interesting train line in Tokyo?")
    await kani.print_stream(stream)

After a stream finishes, its contents will be available as a :class:`.ChatMessage`. You can retrieve the final
message or :class:`.BaseCompletion` with:

.. code-block:: python

    msg = await stream.message()
    completion = await stream.completion()

The final :class:`.ChatMessage` may contain non-yielded tokens (e.g. a request for a function call). If the final
message or completion is requested before the stream is iterated over, the stream manager will consume the entire
stream.

.. tip::
    For compatibility and ease of refactoring, awaiting the stream itself will also return the message, i.e.:

    .. code-block:: python

        msg = await ai.chat_round_stream("What is the airspeed velocity of an unladen swallow?")

    (note the ``await`` that is not present in the above examples). This allows you to refactor your code by changing
    ``chat_round`` to ``chat_round_stream`` without other changes.

    .. code-block:: diff

        - msg = await ai.chat_round("What is the airspeed velocity of an unladen swallow?")
        + msg = await ai.chat_round_stream("What is the airspeed velocity of an unladen swallow?")

Few-Shot Prompting
------------------
Few-shot prompting (AKA in-context learning) is the idea that language models can "learn" the task the user wants
to accomplish through examples provided to it in its prompt.

To few-shot prompt a language model with kani, you can initialize it with an existing chat history. In this example,
we give the model a few-shot prompt in which it translates English to Japanese, and see that it continues to do so
in the chat session despite never being explicitly prompted to do so.

.. code-block:: pycon

    >>> from kani import ChatMessage
    >>> fewshot = [
    ...     ChatMessage.user("thank you"),
    ...     ChatMessage.assistant("arigato"),
    ...     ChatMessage.user("good morning"),
    ...     ChatMessage.assistant("ohayo"),
    ... ]
    >>> ai = Kani(engine, chat_history=fewshot)
    >>> chat_in_terminal(ai, rounds=1)
    USER: crab
    AI: kani

.. tip::
    Passing the fewshot prompt as ``chat_history`` allows kani to manage it as normal - meaning it can slide out of the
    context window. For kani to *always* include the fewshot prompt, use ``always_included_messages``.

Saving & Loading Chats
----------------------
You can save or load a kani's chat state using :meth:`.Kani.save` and :meth:`.Kani.load`. This will dump the state to
a specified JSON file, which you can load into a later kani instance:

.. automethod:: kani.Kani.save
    :noindex:

.. automethod:: kani.Kani.load
    :noindex:

If you'd like more manual control over how you store chat state, there are two attributes you need to save:
:attr:`.Kani.always_included_messages` and :attr:`.Kani.chat_history` (both lists of :class:`.ChatMessage`\ ).

These are `pydantic <https://docs.pydantic.dev/latest/usage/serialization/>`_ models, which you can save and load using
``ChatMessage.model_dump()`` and ``ChatMessage.model_validate()``.

You could, for example, save the chat state to a database and load it when necessary. A common pattern is to save
only the ``chat_history`` and use ``always_included_messages`` as an application-specific prompt.

Next Steps
----------
In the next section, we'll look at subclassing :class:`.Kani` in order to supply functions to the language model.
Then, we'll look at how you can override and/or extend the implementations of kani methods to control each part of
a chat round.
