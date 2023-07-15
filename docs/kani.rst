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

.. code-block:: pythonconsole

    >>> from kani import Kani, chat_in_terminal
    >>> from kani.engines import OpenAIEngine
    >>> api_key = "sk-..."
    >>> engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    >>> ai = Kani(engine)
    >>> chat_in_terminal(ai)
    USER: Hello kani!
    AI: Hello. How can I help you today?
    ^C
    >>> ai.chat_history
    [
        ChatMessage(role=ChatRole.USER, content="Hello kani!"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Hello. How can I help you today?"),
    ]

Engine
------
This table lists the engines built in to kani:

.. todo: engine template here

We won't go too far into implementation details here - if you are interested in implementing your own engine, check out
:doc:`engines` or the :cls:`BaseEngine` API documentation.

Each engine must implement two methods: :meth:`BaseEngine.message_len`, which takes a single :cls:`ChatMessage` and
returns the length of that message, in tokens, and :meth:`BaseEngine.predict`, which is responsible for taking
a list of :cls:`ChatMessage`s and :cls:`AIFunction`s (discussed in the next section) and returning a new
:cls:`Completion <BaseCompletion>`.

These methods are lower-level and used by :cls:`Kani` to manage the chat, but you can also call them yourself.

Kani
----
At a high level, a :cls:`Kani` is responsible for managing a list of :cls:`ChatMessage`s: the chat session associated
with it.





Next Steps
----------
In the next section, we'll look at subclassing :cls:`Kani` in order to supply functions to the language model.
Then, we'll look at how you can override and/or extend the implementations of kani methods to control each part of
a chat round.
