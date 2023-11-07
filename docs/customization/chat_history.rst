Custom Chat History Updates
===========================
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
        # You can override __init__ and track kani-specific state:
        # in this example we keep track of the file we're logging to.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.log_file = open("kani-log.jsonl", "w")

        async def add_to_history(self, message, *args, **kwargs):
            await super().add_to_history(message, *args, **kwargs)
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
