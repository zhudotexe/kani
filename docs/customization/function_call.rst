Handle a Function Call
======================

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

If any of these steps fail, the handler will throw a :exc:`.FunctionCallException`. You might want to override it to
add instrumentation, though:

.. automethod:: kani.Kani.do_function_call
    :noindex:

For example, here's how you might override the method to keep track of how many times a model called a function
during a conversation, and how often it was successful:

.. seealso::

    This example is available in the
    `GitHub repo <https://github.com/zhudotexe/kani/blob/main/examples/3_customization_track_function_calls.py>`__.

.. code-block:: python
    :emphasize-lines: 8-15

    class TrackCallsKani(Kani):
        # You can override __init__ and track kani-specific state:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.successful_calls = collections.Counter()
            self.failed_calls = collections.Counter()

        async def do_function_call(self, call, *args, **kwargs):
            try:
                result = await super().do_function_call(call, *args, **kwargs)
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
