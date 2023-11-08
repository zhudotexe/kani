Handle a Function Call Exception
================================
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

:meth:`.Kani.handle_function_call_exception` controls this behaviour, returning the message to add and whether or not
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
    :emphasize-lines: 2-10

    class CustomExceptionPromptKani(Kani):
        async def handle_function_call_exception(self, call, err, attempt, *args, **kwargs):
            # get the standard retry logic...
            result = await super().handle_function_call_exception(call, err, attempt, *args, **kwargs)
            # but override the returned message with our own
            result.message = ChatMessage.system(
                "The call encountered an error. "
                f"Relay this error message to the user in a sarcastic manner: {err}"
            )
            return result

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
