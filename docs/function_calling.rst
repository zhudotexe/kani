Function Calling
================
Sometimes called "tool usage", function calling gives language models the ability to choose when to call a function you
provide based off its documentation.

With kani, you can write functions in Python and expose them to the model with just one line of code: the
``@ai_function`` decorator.

Step 1: Subclass Kani
---------------------
To create a kani with function calling, make a *subclass* of :class:`.Kani` and write your functions as methods.

For example, you might call an API, perform some math, or retrieve information from the internet - the possibilities
are limitless.

.. code-block:: python
    :emphasize-lines: 7, 9

    from kani import Kani, chat_in_terminal
    from kani.engines.openai import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-4o-mini")

    class MyKani(Kani):
        # step 1: write your methods
        def get_weather(self, location, unit):
            # call some weather API...

    ai = MyKani(engine)
    chat_in_terminal(ai)

.. note::
    AI functions can be synchronous (i.e. ``def``) or asynchronous (``async def``) - kani will automatically await a
    coroutine as needed.

Step 2: Documentation
---------------------
In order for a language model to effectively know what our AI functions do, we need to document them. We do this
inline in the function: through type annotations and the docstring.

The allowed types are:

- Python primitive types (``None``, :class:`bool`, :class:`str`, :class:`int`, :class:`float`)
- an enum (:class:`enum.Enum`)
- a list or dict of the above types (e.g. ``list[str]``, ``dict[str, int]``, ``list[SomeEnum]``)

When the AI calls into the function, kani validates the AI's requested parameters and *guarantees* that the passed
parameters are of the annotated type by the time they reach your code.

Name & Descriptions
^^^^^^^^^^^^^^^^^^^
By default, the function's description will be taken from its docstring, and name from the source.

To specify the descriptions of parameters, you can provide an :class:`.AIParam` annotation using a
:class:`typing.Annotated` type annotation.

For example, you might annotate a parameter ``timezone: str`` with an example, like
``timezone: Annotated[str, AIParam("The IANA time zone, e.g. America/New_York")]``.

Example
^^^^^^^
Now, let's put this all together: let's tell the language model what we expect in the location, and that the unit
should be either fahrenheit or celsius.

.. code-block:: python
    :emphasize-lines: 10, 11, 12, 18, 20, 23

    import enum
    from typing import Annotated

    # don't forget to import AIParam!
    from kani import AIParam, Kani, chat_in_terminal

    # ...

    # for a param with limited choices, define an enum:
    class Unit(enum.Enum):
        FAHRENHEIT = "fahrenheit"
        CELSIUS = "celsius"

    class MyKani(Kani):
        def get_weather(
            self,
            # give the model more information about a parameter by annotating it with AIParam:
            location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
            # or it can determine which of a limited set of options to use from an enum:
            unit: Unit,
        ):
            # add a triple-quoted string immediately after the def to describe the function:
            """Get the current weather in a given location."""
            # call some weather API...

    # ...

.. note::
    Comments (i.e. ``# ...``) aren't given to the language model at all - these are only for your own reference.

Step 3: Register
----------------
The final step once you've defined your method is to register it as an AI function using the ``@ai_function()``
decorator.

Here, you can set some options for how kani should expose your function by passing these keyword args:

.. autofunction:: kani.ai_function
    :noindex:

.. code-block:: python
    :emphasize-lines: 7

    # don't forget to import ai_function!
    from kani import AIParam, Kani, ai_function, chat_in_terminal

    # ...

    class MyKani(Kani):
        @ai_function()
        def get_weather(
            self,
            location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
            unit: Unit,
        ):
            """Get the current weather in a given location."""
            # call some weather API...

    # ...

.. seealso::

    The :func:`.ai_function` API reference.

.. _next_actor:

Next Actor
^^^^^^^^^^
After a function call returns, kani will hand control back to the LM to generate a response by default. If instead
control should be given to the human (i.e. return from the chat round), set ``after=ChatRole.USER``.

.. note::
    If the model calls multiple tools in parallel, the model will be allowed to generate a response if *any* function
    has ``after=ChatRole.ASSISTANT`` (the default) once all function calls are complete.

Complete Example
----------------
Here's the full example of how you might implement a function to get weather that we built in the last few steps:

.. code-block:: python

    import enum
    from typing import Annotated

    from kani import AIParam, Kani, ai_function, chat_in_terminal
    from kani.engines.openai import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-4o-mini")

    class Unit(enum.Enum):
        FAHRENHEIT = "fahrenheit"
        CELSIUS = "celsius"

    class MyKani(Kani):
        @ai_function()
        def get_weather(
            self,
            location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
            unit: Unit,
        ):
            """Get the current weather in a given location."""
            # call some weather API, or just mock it for this example
            degrees = 72 if unit == Unit.FAHRENHEIT else 22
            return f"Weather in {location}: Sunny, {degrees} degrees {unit.value}."

    ai = MyKani(engine)
    chat_in_terminal(ai)

Few-Shot Prompting
------------------
Just as in the last section, we can also few-shot prompt the model to give it examples of how it should call the
functions we define.

When a function returns a result, that result is converted to a string and saved to the chat history. To few-shot
prompt a model, we can mock these returns in the chat history using :meth:`.ChatMessage.function`!

For example, here's how you might prompt the model to give the temperature in both Fahrenheit and Celsius without
the user having to ask:

.. tab:: ToolCall API

    .. code-block:: python

        # build the chat history with examples
        fewshot = [
            ChatMessage.user("What's the weather in Philadelphia?"),
            ChatMessage.assistant(
                content=None,
                # use a walrus operator to save a reference to the tool call here...
                tool_calls=[
                    tc := ToolCall.from_function("get_weather", location="Philadelphia, PA", unit="fahrenheit")
                ],
            ),
            ChatMessage.function(
                "get_weather",
                "Weather in Philadelphia, PA: Partly cloudy, 85 degrees fahrenheit.",
                # ...so this function result knows which call it's responding to
                tc.id
            ),
            # and repeat for the other unit
            ChatMessage.assistant(
                content=None,
                tool_calls=[
                    tc2 := ToolCall.from_function("get_weather", location="Philadelphia, PA", unit="celsius")
                ],
            ),
            ChatMessage.function(
                "get_weather",
                "Weather in Philadelphia, PA: Partly cloudy, 29 degrees celsius.",
                tc2.id
            ),
            ChatMessage.assistant("It's currently 85F (29C) and partly cloudy in Philadelphia."),
        ]
        # and give it to the kani when you initialize it
        ai = MyKani(engine, chat_history=fewshot)

.. tab:: FunctionCall API (deprecated)

    .. code-block:: python

        from kani import ChatMessage, FunctionCall
        fewshot = [
            ChatMessage.user("What's the weather in Philadelphia?"),
            # first, the model should ask for the weather in fahrenheit
            ChatMessage.assistant(
                content=None,
                function_call=FunctionCall.with_args(
                    "get_weather", location="Philadelphia, PA", unit="fahrenheit"
                )
            ),
            # and we mock the function's response to the model
            ChatMessage.function(
                "get_weather",
                "Weather in Philadelphia, PA: Partly cloudy, 85 degrees fahrenheit.",
            ),
            # repeat in celsius
            ChatMessage.assistant(
                content=None,
                function_call=FunctionCall.with_args(
                    "get_weather", location="Philadelphia, PA", unit="celsius"
                )
            ),
            ChatMessage.function(
                "get_weather",
                "Weather in Philadelphia, PA: Partly cloudy, 29 degrees celsius.",
            ),
            # finally, give the result to the user
            ChatMessage.assistant("It's currently 85F (29C) and partly cloudy in Philadelphia."),
        ]
        ai = MyKani(engine, chat_history=fewshot)

.. code-block:: pycon

    >>> chat_in_terminal(ai)
    USER: What's the weather in San Francisco?
    AI: Thinking (get_weather)...
    AI: Thinking (get_weather)...
    AI: It's currently 72F (22C) and sunny in San Francisco.

Few-shot prompts combined with function calls are a powerful tool! For example, you can also specify how a model should
retry functions, vary the parameters it gives, react to function feedback, and more.

Dynamic Functions
-----------------
Rather than statically defining the list of functions a kani can use in a class, you can also pass a list of
:class:`.AIFunction` when you initialize a kani.

The API for the :class:`.AIFunction` class is similar to :func:`.ai_function`.

.. code-block:: python

    def my_cool_function(
        foo: str,
        bar: Annotated[int, AIParam(desc="Some cool parameter.")],
    ):
        """Do some cool things."""
        ...

    functions = [AIFunction(my_cool_function)]
    ai = Kani(engine, functions=functions)

.. _auto_retry:

Retry & Model Feedback
----------------------
If the model makes an error when attempting to call a function (e.g. calling a function that does not exist or
passing params with invalid, non-coercible types) or the function raises an exception, Kani will send the
error in a message to the model by default, allowing it up to *retry_attempts* to correct itself and retry the
call.

.. note::
    If the model calls multiple tools in parallel, the model will be allowed a retry if *any* exception handler
    allows it. This will only count as 1 retry attempt regardless of the number of functions that raised an exception.

In the next section, we'll discuss how to customize this behaviour, along with other parts of the kani interface.

.. _functioncall_v_toolcall:

Internal Representation
-----------------------

.. versionchanged:: v0.6.0

As of Nov 6, 2023, OpenAI added the ability for a single assistant message to request calling multiple functions in
parallel, and wrapped all function calls in a :class:`.ToolCall` wrapper. In order to add support for this in kani while
maintaining backwards compatibility with OSS function calling models, a :class:`.ChatMessage` actually maintains the
following internal representation:

:attr:`.ChatMessage.function_call` is actually an alias for ``ChatMessage.tool_calls[0].function``. If there is more
than one tool call in the message, kani will raise an exception.

A ToolCall is effectively a named wrapper around a :class:`.FunctionCall`, associating the request with a generated
ID so that its response can be linked to the request in future rounds of prompting.
