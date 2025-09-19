Saving & Loading Kani State
===========================
To save and load a :class:`.Kani` object's state, use :meth:`.Kani.save` and :meth:`.Kani.load`.

These methods will save and load the kani's **chat state** to a ``.kani`` or ``.json`` file. You can use this to
save common prompts or log conversations.

.. note::
    The save and load methods only save the **chat state**, not the engine or any functions included in the Kani.
    The chat state includes all ``always_included_messages`` and ``chat_history`` only.

.. automethod:: kani.Kani.save
    :noindex:

.. automethod:: kani.Kani.load
    :noindex:

The .kani File Format
---------------------
What is a ``.kani`` file?

A ``.kani`` file is a self-contained ZIP archive. It contains an ``index.json`` file with the saved chat state.
Additional blobs MAY be saved at ``/blobs/{first 2 characters of hash}/{SHA256}[.suffix]``.

It is safe to change the file extension of a ``.kani`` file to ``.zip`` for manual unzipping and inspection.


For example, let's save the following chat state, with ``kani-multimodal-core`` installed:

.. code-block:: python

    ai = Kani(engine, chat_history=[
        ChatMessage.user([BinaryFilePart.from_file("myfile.pdf"), "What is in this file?"]),
        ChatMessage.assistant("..."),
        ...
        ChatMessage.user([ImagePart.from_file("bird.png")])
    ])
    ai.save("mykani.zip")

The contents of the ``mykani.zip`` archive might look like:

.. code-block:: text

    mykani.zip/
        index.json
        blobs/
            e1/
                e1ab6f04[...].pdf
            f0/
                f01a15c1[...].png
