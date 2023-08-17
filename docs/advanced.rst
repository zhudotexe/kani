Advanced Usage
==============
In this section, we'll look at some of the advanced use cases you can use kani for.
Each of these use cases has an example in `the GitHub repo <https://github.com/zhudotexe/kani/tree/main/examples>`_.

Sub-kanis
---------
When used in conjunction with :doc:`function_calling`, kani can choose when to spawn "sub-kani" - self-contained
"agents" capable of performing their own tasks, then reporting to the parent with their results.

For example, you might have the parent kani use a cheaper, faster model - but with the tradeoff that that model has a
smaller context length. If you need it to perform a task that requires more context, you can spawn a sub-kani using
a more expensive, slower model with a larger context.

.. caution::
    Be careful when creating a new kani instance with an existing kani's chat history!
    If you pass an old kani's chat history to a new kani without copying it, the same list will be mutated.

    Use ``newkani = Kani(..., chat_history=oldkani.chat_history.copy())`` to pass a copy.

    Index slicing (as shown in the example below) also creates a copy.

.. code-block:: python

    class KaniWithAISummarization(Kani):
        @ai_function()
        async def summarize_conversation(self):
            """Get the summary of the conversation so far."""
            # in this AI Function, we can spawn a sub-kani with a model that can handle
            # longer contexts, since the conversation may be longer than the fast model's
            # context window
            long_context_engine = OpenAIEngine(api_key, model="gpt-4-32k")
            # copy the parent's chat history to the child, except the last user message
            # and the function call ([:-2])
            sub_kani = Kani(long_context_engine, chat_history=self.chat_history[:-2])
            # then we ask it to summarize the whole thing, and return the result to the parent
            return await sub_kani.chat_round_str("Please summarize the conversation so far.")

Of course, the sub-kani you spawn doesn't have to be a vanilla kani - you could imagine having multiple different
kani types with different sets of functions or engines, each capable of performing their own specialized tasks.

.. code-block:: pycon

    >>> chat_in_terminal(KaniWithAISummarization(engine))
    USER: Tell me about trains.
    AI: Trains are a mode of long-distance transport...

    [Multiple turns of conversation.]

    USER: Summarize the conversation.
    AI: Thinking (summarize_conversation)...
    AI: Our conversation began with a general overview about trains, their history, types,
    benefits, and how railway systems work around the world. We then moved onto discussing
    the best high-speed rail lines around the world, with a specific focus on the Japanese
    train system.

    Your interest in trainspotting in Tokyo led to the sharing of some popular locations
    in the city and a discussion about unique, non-standard train lines in Japan, including
    lines that go through a tunnel of trees.

    After exploring the topic of trains in Japan, we shifted to talk about lesser-known
    attractions in Japan, which led to detailing a potential summer itinerary for a trip
    starting in Tokyo and heading south.

Retrieval
---------
Retrieval is the idea that models can be augmented with an external factual database that they can *retrieve*
information from, allowing them to access more relevant and up-to-date information without having to train.

In this example, we demonstrate how kani's function calling can be used to retrieve information from a data source
like Wikipedia.

.. code-block:: python

    # let's define a client using kani's BaseClient:
    class WikipediaClient(BaseClient):
        SERVICE_BASE = "https://en.wikipedia.org/w/api.php"


    class WikipediaRetrievalKani(Kani):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.wikipedia_client = WikipediaClient()

        @ai_function()
        async def wikipedia(
            self,
            title: Annotated[
                str,
                AIParam(desc='The article title on Wikipedia, e.g. "Train_station".')
            ],
        ):
            """Get additional information about a topic from Wikipedia."""
            resp = await self.wikipedia_client.get(
                "/",
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "titles": title,
                    "explaintext": 1,
                    "formatversion": 2,
                },
            )
            page = resp["query"]["pages"][0]
            if extract := page.get("extract"):
                return extract
            return f"The page {title!r} does not exist on Wikipedia."

        @ai_function()
        async def search(self, query: str):
            """Find titles of Wikipedia articles similar to the given query."""
            resp = await self.wikipedia_client.get(
                "/",
                params={"action": "opensearch", "format": "json", "search": query}
            )
            return json.dumps(resp[1])

.. code-block:: pycon

    >>> chat_in_terminal(WikipediaRetrievalKani(engine))
    USER: Tell me about the Yamanote line in Tokyo.
    AI: Thinking (search)...
    AI: Thinking (wikipedia)...
    AI: The Yamanote Line is a loop service in Tokyo, Japan...

.. caution::

    Wikipedia articles might be longer than can fit in the model's context window. Try combining this with the sub-kani
    summarization example above to build a powerful retrieval agent!

    You may also use ``@ai_function(auto_truncate=...)`` if truncating the response is acceptable.

Hosting kani Online
-------------------
What if you want to host a web service that allows its users to chat with kani? In this example, we show how you can
allow users to connect to a kani hosted on a webserver using a WebSocket connection. Since kani supports asyncio and
is built with parallelization in mind, you can have as many people as you want connect at once!

We'll use `FastAPI <https://fastapi.tiangolo.com/>`_ to run this webserver. To connect to it, you can use any client
that supports WebSockets, like `Insomnia <https://insomnia.rest/>`_.

.. code-block:: python

    # normally, reusing an OpenAIEngine isn't recommended
    # but since we don't define any @ai_functions, it's okay
    engine = OpenAIEngine(api_key, model="gpt-3.5-turbo")
    app = FastAPI()

    @app.websocket("/chat")
    async def kani_chat(websocket: WebSocket):
        # accept the websocket and initialize a kani for the connection
        await websocket.accept()
        ai = Kani(engine)
        # take string messages and send string responses
        while True:
            try:
                data = await websocket.receive_text()
                resp = await ai.chat_round_str(data)
                await websocket.send_text(resp)
            # until the client disconnects
            except WebSocketDisconnect:
                return


    @app.on_event("shutdown")
    async def cleanup_kani():
        """When the application shuts down, cleanly close the kani engine."""
        await engine.close()

Now, you can run the service and connect to it (by default, ``uvicorn`` binds to ``127.0.0.1:8000``).

.. code-block:: pycon

    >>> uvicorn.run(app)

.. image:: _static/5_advanced_api.png
    :align: center
    :width: 600

.. tip::

    In a real production environment, you might want to send JSON payloads over the WebSocket rather than raw strings.

Function "Loadouts": Polymorphism & Mixins
------------------------------------------
What if you have some common AI functions that you want to share with multiple kani? For example, what if you've
written a solid calculator suite you want to give to two different kani - one with web retrieval and one without.

Since kani are implemented in pure Python, you can accomplish this with polymorphism!

Base Class
^^^^^^^^^^
One method is to make one of the kani a base, then subclass it to extend its functionality:

.. code-block:: python

    class BaseKani(Kani):
        @ai_function()
        def calculate(self, expr: str):
            ...

    class ChildKani(BaseKani):
        @ai_function()
        def search(self, query: str):
            ...

In this example, the ``ChildKani`` has access to both ``search`` *and* ``calculate``.

Mixins
^^^^^^
But in some cases, you won't necessarily have a single base to extend, and you might want to share multiple different
modular "loadouts" of functions. ``@ai_function``\ s don't have to be defined in :class:`.Kani` classes, so in this
case, you can use a mixin!

.. code-block:: python

    # note: the mixin isn't a subclass of Kani!
    class CalculatorMixin:
        @ai_function()
        def add(self, left: float, right: float):
            return left + right

        @ai_function()
        def mul(self, left: float, right: float):
            return left * right

    # reuse it to give a loadout of functions to multiple kani!
    class MyKani(CalculatorMixin, Kani):
        @ai_function()
        def search(self, query: str):
            ...

    class SomeOtherKani(CalculatorMixin, Kani):
        @ai_function()
        def music(self, song: str):
            ...

In this example, both the kani (``MyKani`` and ``SomeOtherKani``) have access to ``add`` and ``mul`` in addition to
the functions defined in their class body.

Just as in normal Python, you can inherit from multiple mixins. You can use this to build kani with modular sets of
functionality!
