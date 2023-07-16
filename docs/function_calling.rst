Function Calling
================
Sometimes called "tool usage", function calling gives language models the ability to choose when to call a function you
provide based off its documentation.

With kani, you can write functions in Python and expose them to the model with just one line of code: the
``@ai_function`` decorator.

Step 1: Subclass Kani
---------------------
To create a kani with function calling, make a *subclass* of :class:`Kani` and write your functions as methods.

For example, you might call an API, perform some math, or retrieve information from the internet - the possibilities
are limitless.

.. code-block:: python

    from kani import Kani, chat_in_terminal
    from kani.engines import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")

    class MyKani(Kani):
        # step 1: write your methods
        def get_weather(location, unit):
            # call some weather API...

    ai = MyKani(engine)
    chat_in_terminal(ai)

.. note::

    AI functions can be synchronous or asynchronous - kani will automatically await a coroutine as needed.

Step 2: Documentation
---------------------
In order for a language model to effectively know what our AI functions do, we need to document them. We do this
inline in the function: through type annotations and the docstring.

The allowed types are:

- Python primitive types (``None``, :class:`bool`, :class:`str`, :class:`int`, :class:`float`)
- an enum (subclass of ``enum.Enum``)
- a list or dict of the above types (e.g. ``list[str]``, ``dict[str, int]``, ``list[SomeEnum]``)

When the AI calls into the function, kani validates the AI's requested parameters and *guarantees* that the passed
parameters are of the annotated type by the time they reach your code.

Name & Descriptions
^^^^^^^^^^^^^^^^^^^
By default, the function's description will be taken from its docstring, and name from the source.

To specify the descriptions of parameters, you can provide an :class:`AIParam` annotation using a
:class:`typing.Annotated` type annotation.

For example, you might annotate a parameter ``timezone: str`` with an example, like
``timezone: Annotated[str, AIParam("The IANA time zone, e.g. America/New_York")]``.

Example
^^^^^^^
Now, let's put this all together: let's tell the language model what we expect in the location, and that the unit
should be either fahrenheit or celsius.

.. code-block:: python

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
            # we can give the model more information about a parameter by annotating it with AIParam:
            location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
            # or it can determine which of a limited set of options to use from an enum:
            unit: Unit,
        ):
            # finally, add a triple-quoted string immediately after the def to describe the function:
            """Get the current weather in a given location."""
            # call some weather API...

    # ...

Step 3: Register
----------------
The final step once you've defined your method is to register it as an AI function using the ``@ai_function()``
decorator.

Here, you can set some options for how kani should expose your function by passing these keyword args:

- ``after``: Who should speak next after the function call completes (see :ref:`next_actor`). Defaults to the model.
- ``name``: The name of the function (defaults to the name of the function in source code)
- ``desc``: The function's description (defaults to the function's docstring)
- ``auto_retry``: Whether the model should retry calling the function if it gets it wrong (see :ref:`auto_retry`).
- ``json_schema``: A JSON Schema document describing the function's parameters. By default, kani will automatically
  generate one, but this can be helpful for overriding it in any tricky cases.

.. code-block:: python

    # don't forget to import ai_function!
    from kani import AIParam, Kani, ai_function, chat_in_terminal

    # ...

    class MyKani(Kani):
        @ai_function()
        def get_weather(
            location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
            unit: Unit,
        ):
            """Get the current weather in a given location."""
            # call some weather API...

    # ...

.. seealso::

    The :func:`ai_function` API reference.

Next Actor
^^^^^^^^^^
After a function call returns, kani will hand control back to the LM to generate a response by default. If instead
control should be given to the human (i.e. return from the chat round), set ``after=ChatRole.USER``.

Example
-------
Here's the full example of how you might implement a function to get weather that we built in the last few steps:

.. code-block:: python

    from kani import AIParam, Kani, ai_function, chat_in_terminal
    from kani.engines import OpenAIEngine

    api_key = "sk-..."
    engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")

    class Unit(enum.Enum):
        FAHRENHEIT = "fahrenheit"
        CELSIUS = "celsius"

    class MyKani(Kani):
        @ai_function()
        def get_weather(
            location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
            unit: Unit,
        ):
            """Get the current weather in a given location."""
            # call some weather API, or just mock it for this example
            degrees = 72 if unit == Unit.FAHRENHEIT else 22
            return f"Weather in {location}: Sunny, {degrees} degrees {unit.value}."

    ai = MyKani(engine)
    chat_in_terminal(ai)

Dynamic Functions
-----------------
Rather than statically defining the list of functions a kani can use in a class, you can also pass a list of
:class:`AIFunction` when you initialize a kani.

The API for the :class:`AIFunction` class is similar to :func:`ai_function`.

.. code-block:: python

    def my_cool_function(foo: str, bar: int):
        ...

    functions = [AIFunction(my_cool_function)]
    ai = Kani(engine, functions=functions)

Retry & Model Feedback
----------------------
If the model makes an error when attempting to call a function (e.g. calling a function that does not exist or
passing params with invalid, non-coercible types) or the function raises an exception, Kani will send the
error in a system message to the model by default, allowing it up to *retry_attempts* to correct itself and retry the
call.

In the next section, we'll discuss how to customize this behaviour, along with other parts of the kani interface.
