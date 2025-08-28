Message Extras (Metadata)
=========================

Message Extras are a way to attach engine-specific metadata to a :class:`.ChatMessage` object. It is a dictionary
that remains attached to the message for the duration of its lifetime.

.. warning::

    Don't confuse Message Extras with :doc:`messageparts`. Message Extras are used for engine-specific metadata
    about a certain message (e.g., an internal ID or detailed engine-specific usage data), whereas Message Parts are
    used for engine-agnostic inputs that require richer representation than a string (e.g., a multimodal input or a
    hidden chain of thought).

You can use Message Extras to store additional semi-structured information for your own downstream use. Certain
engines may also use Message Extras to return engine-specific metadata.

.. autoattribute:: kani.ChatMessage.extra
    :noindex:

The :class:`.MessagePart` class also contains a similar attribute for the same purpose:

.. autoattribute:: kani.MessagePart.extra
    :noindex:

Example (OpenAI)
----------------
One example of using Message Extras is to retrieve an OpenAI-specific usage object. Although kani returns basic
``prompt_tokens`` and ``completion_tokens`` usage attributes with most completions, OpenAI completions contain much
more detailed usage:

.. code-block:: json

    "usage": {
        "prompt_tokens": 1117,
        "completion_tokens": 46,
        "total_tokens": 1163,
        "prompt_tokens_details": {
          "cached_tokens": 0,
          "audio_tokens": 0
        },
        "completion_tokens_details": {
          "reasoning_tokens": 0,
          "audio_tokens": 0,
          "accepted_prediction_tokens": 0,
          "rejected_prediction_tokens": 0
        }
    }

However, this detailed usage is only available when using the :class:`.OpenAIEngine`. In order to store this detailed
usage data without interfering with other engines, kani saves this usage object to the ``"openai_usage"`` message extra.

.. autoclass:: kani.engines.openai.OpenAIEngine
    :noindex:

So, we can access this usage like so:

.. code-block:: python

    from kani.engines.openai import OpenAIEngine
    from kani import Kani

    async def detailed_usage():
        engine = OpenAIEngine(model="gpt-5-nano")
        ai = Kani(engine)
        msg = await ai.chat_round("How many 'o's are in 'pneumonoultramicroscopicsilicovolcanoconiosis'?")
        print(msg.text)
        print(msg.extra.get("openai_usage"))

.. code-block:: pycon

    >>> import asyncio
    >>> asyncio.run(detailed_usage())
    9
    Reason: Break it into segments—pneumo, noultra, micro, scopic, silico, volcano, coniosis—each contains 1,1,1,1,1,2,2 o’s respectively, totaling 9.

    CompletionUsage(
        completion_tokens=1663,
        prompt_tokens=30,
        total_tokens=1693,
        completion_tokens_details=CompletionTokensDetails(
            accepted_prediction_tokens=0,
            audio_tokens=0,
            reasoning_tokens=1600,
            rejected_prediction_tokens=0
        ),
        prompt_tokens_details=PromptTokensDetails(
            audio_tokens=0,
            cached_tokens=0
        )
    )


Best Practices
--------------

* A given extra MAY NOT be present on a ChatMessage returned by different engines.
* Downstream code SHOULD NOT rely on the presence of a certain extra, but MAY conditionally check for the presence
  of certain extras for logging purposes. Reliance on certain extras tightly couples code with a certain engine.
* An engine SHOULD NOT rely on an extra it set in a past round being present in future rounds.
