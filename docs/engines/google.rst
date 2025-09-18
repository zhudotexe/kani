GoogleAIEngine
==============
The :class:`.GoogleAIEngine` is used to make requests to the Google AI Studio API.

**TL;DR**

.. code-block:: python

    # see https://ai.google.dev/gemini-api/docs/models for a list of model IDs
    from kani.engines.google import GoogleAIEngine
    engine = GoogleAIEngine(api_key=os.getenv("GEMINI_API_KEY"), model="gemini-2.5-flash")

Reference
---------

.. autoclass:: kani.engines.google.GoogleAIEngine
    :noindex:

Notes
-----

**Large Multimodal File Handling**: When a multimodal message part exceeds the ``multimodal_upload_bytes_threshold``
set in the ``GoogleAIEngine``'s constructor, kani will upload the file to the Files API. This allows the file to be
reused across multiple requests without uploading the full body with each request.

Recipes
-------

Thinking
^^^^^^^^

.. code-block:: python

    import google.genai.types as gai
    from kani.engines.google import GoogleAIEngine

    # see https://ai.google.dev/gemini-api/docs/thinking for the thinking_budget explanation
    thinking_engine = GoogleAIEngine(..., thinking_config=gai.ThinkingConfig(thinking_budget=-1, include_thoughts=True))

Server-Side Tools
^^^^^^^^^^^^^^^^^
To enable server-side tools, you pass them as additional arguments to the ``tools`` API argument.
You can do this by overriding ``GoogleAIEngine._prepare_request``.

.. code-block:: python

    import google.genai.types as gai
    from kani.engines.google import GoogleAIEngine

    class GoogleAIServersideToolsEngine(GoogleAIEngine):
        def __init__(self, *args, additional_tools: list[gai.Tool] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.additional_tools = additional_tools or []

        # override prepare_request to inject serverside tool configs
        def _prepare_request(self, messages, functions):
            generation_config, prompt_msgs = super()._prepare_request(messages, functions)
            if self.additional_tools:
                generation_config.setdefault("tools", [])
                generation_config["tools"].extend(self.additional_tools)
            return generation_config, prompt_msgs

    web_search_engine = GoogleAIServersideToolsEngine(..., additional_tools=[
        gai.Tool(url_context=gai.UrlContext()),
        gai.Tool(google_search=gai.GoogleSearch()),
    ])