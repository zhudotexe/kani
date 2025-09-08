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
