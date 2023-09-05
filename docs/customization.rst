Customization
=============
Now that you're familiar with subclassing :class:`.Kani` in order to implement function calling, we can take a look at
the other parts you can customize.

kani is built on the philosophy that every part should be hackable. To accomplish this, kani has a set of overridable
methods you can override in a subclass. This page documents what these methods do by default, and why you might want
to override them.

Build the Chat Prompt
---------------------
Modern language models work by generating a *continuation* to a *prompt*. The prompt contains all the context-specific
knowledge the model has access to; if it's not in the prompt, it can only use its pretraining data.

While we would love to pass the entire chat history in the prompt, models also contain a *token limit* - the maximum
size of the prompt we can give it at one time.

.. hint:: "token limit" is also known as "context size".

Since chats can be longer than a model's token limit, we have to decide which parts to keep and which parts to omit,
creating a sliding window of memory for the LM.

.. important::

    Language models can't remember what happened in a conversation beyond their token limit. Making them do so is a hot
    area of research!

By default, kani includes the **system prompt** and any messages specified as **always include** (in the initializer),
then as many messages as possible fit in the remaining token limit, prioritizing later messages.

.. todo: figure demonstrating this

To override this behaviour, override :meth:`.Kani.get_prompt` in your subclass:

.. automethod:: kani.Kani.get_prompt
    :noindex:

For example, here's how you might override the behaviour to only include the most recent 4 messages
(omitting earlier ones to fit in the token length if necessary) and any always included messages:

.. seealso::

    This example is available in the
    `GitHub repo <https://github.com/zhudotexe/kani/blob/main/examples/3_customization_last_four.py>`__.

.. code-block:: python

    class LastFourKani(Kani):
        async def get_prompt(self):
            # calculate how many tokens we have for the prompt, accounting for the system prompt,
            # always_included_messages, any tokens reserved by the engine, and the response
            remaining = self.max_context_size - self.always_len
            # working backwards through history...
            messages = []
            for message in reversed(self.chat_history[-4:]):
                # if the message fits in the space we have remaining...
                message_len = self.message_token_len(message)
                remaining -= message_len
                if remaining > 0:
                    # add it to the returned prompt!
                    messages.insert(0, message)
                else:
                    break
            return self.always_included_messages + messages

Chatting with this kani, we can see that it loses any memory of what happened more than 4 messages (2 rounds) ago:

.. code-block:: pycon

    >>> chat_in_terminal(LastFourKani(engine))
    USER: Hi kani! My name is Andrew.
    AI: Hello Andrew! How can I assist you today?

    USER: What does "kani" mean in Japanese?
    AI: "Kani" in Japanese means "Crab".

    USER: How do you pronounce it?
    AI: Kani is pronounced as "kah-nee" in Japanese.

    USER: What is my name?
    AI: As an AI, I don't have access to personal data about individuals unless it has
    been shared with me in the course of our conversation. I'm designed to respect user
    privacy and confidentiality.

Custom Chat History Updates
---------------------------
In many cases, you might want to implement your own logic whenever the chat history is updated - whether that's through
the user querying the model, the model's response, a function call, or any other scenario. For example, you might
want to log each message to an external database or hydrate an external vector database.

By default, kani tracks the entire chat history in the :attr:`.Kani.chat_history` attribute and appends all new messages
to it through the :meth:`.Kani.add_to_history` method. We don't recommend changing the default behaviour, but you can
override this method to add your own logic!

.. automethod:: kani.Kani.add_to_history
    :noindex:

For example, here's how you might extend :meth:`.Kani.add_to_history` to log every message to a JSONL file:

.. seealso::

    This example is available in the
    `GitHub repo <https://github.com/zhudotexe/kani/blob/main/examples/3_customization_log_messages.py>`__.

.. hint::
    kani's :class:`.ChatMessage`\ s are `Pydantic models <https://docs.pydantic.dev/latest/usage/models/>`_
    under the hood - which means they come with utilities for serialization and deserialization!

.. code-block:: python

    class LogMessagesKani(Kani):
        # You can overload __init__ and track kani-specific state:
        # in this example we keep track of the file we're logging to.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.log_file = open("kani-log.jsonl", "w")

        async def add_to_history(self, message):
            await super().add_to_history(message)
            self.log_file.write(message.model_dump_json())
            self.log_file.write("\n")

If we chat with this kani and then read ``kani-log.jsonl``, we can see that it logs each message:

.. code-block:: console

    $ python 3_customization_log_messages.py
    USER: Hello kani!
    AI: Hello! How can I assist you today?
    USER: Just saying hi!
    AI: Hi! It's great to have you here.
    ^C
    $ cat kani-log.jsonl
    {"role":"user","content":"Hello kani!","name":null,"function_call":null}
    {"role":"assistant","content":"Hello! How can I assist you today?","name":null,"function_call":null}
    {"role":"user","content":"Just saying hi!","name":null,"function_call":null}
    {"role":"assistant","content":"Hi! It's great to have you here.","name":null,"function_call":null}

This kind of logging isn't just limited to chatting - this example will also log any function calls and retries.

.. _do_function_call:

Handle a Function Call
----------------------

.. note:: This functionality is only available when using :meth:`.Kani.full_round`.

When a model predicts that it should use a function, it will request a :class:`.FunctionCall`. It is then kani's
responsibility to turn the requested function call into a real call to a Python method.

By default, you probably won't want to change the implementation of :meth:`.Kani.do_function_call`, which does a couple
things:

1. Get the right Python function by name.
2. Parse the requested arguments into a Python dictionary and validate its types.
3. Call the Python function.
4. Append a new :class:`.ChatMessage` with the result of the function.
5. Return control to the model or the user.

If any of these steps fail, the handler will throw a :exc:`.FunctionCallException`. You might want to overload it to
add instrumentation, though:

.. automethod:: kani.Kani.do_function_call
    :noindex:

For example, here's how you might overload the method to keep track of how many times a model called a function
during a conversation, and how often it was successful:

.. seealso::

    This example is available in the
    `GitHub repo <https://github.com/zhudotexe/kani/blob/main/examples/3_customization_track_function_calls.py>`__.

.. code-block:: python
    :emphasize-lines: 8-15

    class TrackCallsKani(Kani):
        # You can overload __init__ and track kani-specific state:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.successful_calls = collections.Counter()
            self.failed_calls = collections.Counter()

        async def do_function_call(self, call):
            try:
                result = await super().do_function_call(call)
                self.successful_calls[call.name] += 1
                return result
            except FunctionCallException:
                self.failed_calls[call.name] += 1
                raise

        # Let's give the model some functions to work with:
        @ai_function()
        def get_time(self):
            """Get the current time in the user's time zone."""
            # oh no! the clock is broken!
            raise RuntimeError("The time API is currently offline. Please try using `get_date_and_time`.")

        @ai_function()
        def get_date_and_time(self):
            """Get the current day and time in the user's time zone."""
            return str(datetime.datetime.now())

Chatting with this kani, we can see how it retries the failed call, and how we log the attempts:

.. code-block:: pycon

    >>> chat_in_terminal(TrackCallsKani(engine), rounds=1)
    USER: What time is it?
    AI: Thinking (get_time)...
    AI: Thinking (get_date_and_time)...
    AI: The current time is 22:42.
    >>> ai.successful_calls
    Counter({'get_date_and_time': 1})
    >>> ai.failed_calls
    Counter({'get_time': 1})

.. _handle_function_call_exception:

Handle a Function Call Exception
--------------------------------
.. note:: This functionality is only available when using :meth:`.Kani.full_round`.

Above, we show how you can instrument a function call. But when a function call goes wrong, what happens?

A requested function call can error out for a variety of reasons:

- The requested function doesn't exist and the model hallucinated it (:exc:`.NoSuchFunction`)
- The function exists, but the model hallucinated parameters that don't exist (:exc:`.WrappedCallException` around
  :exc:`TypeError`)
- The parameter names all exist, but the model got the data types wrong or didn't provide some
  (:exc:`.WrappedCallException` around :exc:`TypeError` or
  `ValidationError <https://docs.pydantic.dev/latest/errors/validation_errors/>`_)
- The Python function raised an exception (:exc:`.WrappedCallException`)

By default, kani will add a :class:`.ChatMessage` to the chat history, giving the model feedback
on what occurred. The model can then retry the call up to *retry_attempts* times.

:meth:`.Kani.handle_function_call_exception` controls this behaviour, adding the message and returning whether or not
the model should be allowed to retry. By overriding this method, you can control the error prompt, log the error, or
implement custom retry logic.

The default prompts are:

- :exc:`.NoSuchFunction`: ``ChatMessage.system(f"The function {err.name!r} is not defined. Only use the provided
  functions.")``
- otherwise: ``ChatMessage.function(call.name, str(err))``

.. automethod:: kani.Kani.handle_function_call_exception
    :noindex:

Here's an example of providing custom prompts on an exception:

.. seealso::

    This example is available in the
    `GitHub repo <https://github.com/zhudotexe/kani/blob/main/examples/3_customization_custom_exception_prompt.py>`__.

.. code-block:: python
    :emphasize-lines: 2-7

    class CustomExceptionPromptKani(Kani):
        async def handle_function_call_exception(self, call, err, attempt):
            self.chat_history.append(ChatMessage.system(
                "The call encountered an error. "
                f"Relay this error message to the user in a sarcastic manner: {err}"
            ))
            return attempt < self.retry_attempts and err.retry

        @ai_function()
        def get_time(self):
            """Get the current time in the user's time zone."""
            raise RuntimeError("The time API is currently offline (error 0xDEADBEEF).")

If we chat with this kani, we can see how it follows the instructions in the error prompt:

.. code-block:: pycon

    >>> chat_in_terminal(CustomExceptionPromptKani(engine))
    USER: What time is it?
    AI: Thinking (get_time)...
    AI: Well, it seems like our handy-dandy time API decided to take a coffee break.
    It's currently offline, lounging about with an error code of 0xDEADBEEF.
    Guess we'll have to rely on the old-fashioned way of checking the time!

Next, we'll look at the engines available and how you can implement additional models.
