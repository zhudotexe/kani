Message Parts
=============
In some advanced use cases, :attr:`.ChatMessage.content` may be a tuple of :class:`.MessagePart` or ``str`` rather than
a string. ChatMessage exposes :attr:`.ChatMessage.text` (always a string or None) and :attr:`.ChatMessage.parts` (always
a list of message parts), which we recommend using instead of :attr:`.ChatMessage.content`. These properties are
dynamically generated based on the underlying content, and it is safe to mix messages with different content types in a
single Kani.

Generally, message part classes are *defined* by the engine, and *consumed* by the developer. Message parts can be used
in any role's message - for example, you might use a message part in an assistant message to separate out a chain of
thought from a user reply, or in a user message to supply an image to a multimodal model.

Let's say you wanted your model to perform a chain-of-thought before replying to the user, but didn't want that
chain of thought to be visible in its output. In the example below, we'll define a hidden *ThoughtPart* that contains
this data, and have our engine use this when building the prompt. This may be a little bit contrived, but hopefully
it demonstrates how to use the MessagePart interface.

.. tip::

    MessageParts are commonly used for multimodal inputs: for example, both the :class:`.OpenAIEngine` and
    :class:`.AnthropicEngine` support image inputs using ``kani-multimodal-core``\ 's
    :class:`~kani.ext.multimodal_core.ImagePart`.

.. warning::

    Don't confuse Message Parts with :doc:`message_extras`. Message Parts are used for engine-agnostic inputs
    that require richer representation than a string, whereas Message Extras are used for engine-specific metadata
    about a certain message (e.g., an internal ID or detailed engine-specific usage data).

Defining a MessagePart
----------------------
To define a :class:`.MessagePart`, you should create a new class that inherits from :class:`.MessagePart`.

Under the hood, a MessagePart is a Pydantic model, so to add attributes to your part, you can define them as
type-annotated class variables. For example, to add a ``data`` attribute with a string type, you can define
``data: str``.

Next, you'll want to define the ``__str__()`` method, which is how kani defines the behaviour when your message part
is cast to a string. This happens in a couple places:

1. The message part is displayed to the user through a string-casting method (e.g. :meth:`.Kani.full_round_str`)
2. The user is using the :func:`.chat_in_terminal` development utility
3. The message was provided to an engine that does not support the class (e.g. an image part in a text-only model).

The default behaviour is to transform the part to a Python-esque representation of its data (e.g.
``<ThoughtPart data="...">``) and log a warning. You can override this method to specify the canonical string
representation of your message part.

.. code-block:: python

    from kani import MessagePart

    class ThoughtPart(MessagePart):
        # hold some string data that is the chain-of-thought
        data: str

        # when stringified, this part should be hidden from the user/any engine that does not explicitly support it
        def __str__(self):
            return ""


When you define a MessagePart, kani will automatically register serialization and deserialization schemes for your
class, so that all MessageParts are compatible with :meth:`.Kani.save` and :meth:`.Kani.load`.

.. warning::
    If you change the attributes or location of the class definition, old data may fail to be loaded into a new class.

Using MessageParts in Engines
-----------------------------
Once we define a message part, we need to tell the engine how to use it. Since message parts are designed to provide
engine-specific metadata while maintaining cross-engine compatibility, engines should check for the classes they know
and cast unknown classes to a string. This lets parts control how they will be serialized in engines that don't natively
support them.

A common, but not necessary, pattern is to write an engine that *wraps* other engines. This wrapper engine acts as a
translation layer for specific message parts, eventually compiling a single string (or list of strings) for a base
engine (e.g. LlamaEngine) to consume.

Here's the implementation for our chain-of-thought example. Note how we translate the parts by building a new message
with a string content composed of parts in ``translate_message``, then use that method in our implementations of
``prompt_len`` and ``predict``.

.. code-block:: python

    from kani import AIFunction, ChatMessage, MessagePart
    from kani.engines import WrapperEngine, Completion

    class ChainOfThoughtEngine(WrapperEngine):
        @staticmethod
        def translate_message(message: ChatMessage) -> ChatMessage:
            """Translate a input message into a simple string-only message to pass to the underlying engine."""
            content = ""
            # iterate over parts: if it's a thought, place its data in the content; otherwise prefix it with "Answer: "
            for part in message.parts:
                if isinstance(part, ThoughtPart):
                    content += f"{part.data}\n"
                else:
                    content += f"Answer: {str(part)}"
            # return the translated message
            return message.copy_with(content=content.strip())

        # === BaseEngine interface ===
        async def predict(
            self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
        ) -> Completion:
            # translate the messages
            translated_messages = [self.translate_message(m) for m in messages]

            # generate a completion using the underlying engine given those messages
            result = await self.engine.predict(translated_messages, functions, **hyperparams)

            # parse the string-completion back into parts - with some checks to make sure the model actually output the
            # right format
            text = result.message.text
            if "Answer:" in text:
                thought, answer = text.split("Answer:", 1)
                message_with_parts = result.message.copy_with(parts=[ThoughtPart(data=thought.strip()), answer.strip()])
            else:
                message_with_parts = result.message

            # we'll print the result so you can see the parts, though a real engine wouldn't want to
            print("Model response with parts:", message_with_parts.parts)
            # and return the modified answer
            return Completion(
                message=message_with_parts,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )

Now, we can use this engine by passing an underlying engine to it, prompt our model to follow our chain-of-thought
format, and see how it works!

.. code-block:: python

    cot_engine = ChainOfThoughtEngine(OpenAIEngine())
    ai = Kani(
        cot_engine,
        system_prompt=(
            "When answering the user, think step by step. Output your thoughts first, "
            'then output the final answer on its own line, in the format "Answer: {answer}".'
        ),
    )
    chat_in_terminal(ai)

    # USER: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls.
    # How many tennis balls does he have now?
    #
    # Model response with parts:
    #   [ThoughtPart(data='Roger already has 5 tennis balls. He buys 2 more cans of tennis balls, and each can has 3 tennis balls. \n\nTo find out how many tennis balls he has now, we need to multiply the number of cans with the number of tennis balls in each can. Since he bought 2 cans, we multiply 2 by 3:\n\n2 cans * 3 tennis balls per can = 6 tennis balls from the cans\n\nNext, we add the number of tennis balls he already had:\n\n5 tennis balls + 6 tennis balls = 11 tennis balls'),
    #   'Roger now has 11 tennis balls.']
    #
    # AI: Roger now has 11 tennis balls.

See how the engine splits up the model's response into two parts: the thought and the displayed answer. In the
:func:`.chat_in_terminal` development utility, we only display the displayed answer, but in a larger application you'd
get the full :class:`.ChatMessage`, and you could iterate over the parts to handle displaying the thought with your own
application logic.

If you switch to a different base engine and inject the chat history, the base engine won't see the thoughts, since
the default stringification behaviour is to return the empty string. This means that you can build complex engine
behaviour that won't interfere with other engines.

Now you can use any message part you can think of - and you can create user messages with parts too.

Using MessageParts in Tools
---------------------------
What if an AI function returns multimodal content? By default, Kani applies certain transformations to the return value
of an AIFunction in order:

* if it is a :class:`.ChatMessage`, do not modify it
* if it is a list of :class:`.MessagePart`, do not modify it
* if it is a JSON-serializable Python dict, serialize it to JSON
* if it is a Pydantic model, serialize it to JSON
* otherwise, cast it to a string

Thus, in order to return multimodal content, the AIFunction must return a FUNCTION-role chat message. For example, the
following code snippet defines a function which returns an image:

.. code-block:: python

    class MyKani(Kani):
        @ai_function()
        async def get_image(self, url: str):
            """Download the image at a certain URL, and view it."""
            return [await ImagePart.from_url(url)]


However, most API-based LLMs currently do not support multimodal returns from tools.

Saving & Loading MessageParts
-----------------------------
By default, kani will register a serializer and deserializer for each MessagePart you define, which will recursively
save every public attribute in your MessagePart in JSON format. If all of your data is JSON-serializable, great!
You don't need to do anything extra.

Certain MessageParts, however, may store non-JSON-serializable data, like a large binary file for a multimodal model.
In order to correctly save the state of these MessageParts, you must implement a custom
`serializer <https://docs.pydantic.dev/latest/concepts/serialization/#custom-serializers>`_ and
`validator <https://docs.pydantic.dev/latest/concepts/validators/#model-validators>`_.

You might, for example, save raw binary in Base64 format. When saving large binary objects, you should check for the
presence of ``kani.utils.saveload.SAVELOAD_CONTEXT_KEY`` in the serialization/validation context. If it is present,
you may use the :class:`.KaniZipSaveContext` to save large blobs to a separate location in the save archive.
This has the benefit of keeping the main chat log human-readable and the file size small.

See :doc:`saveload` for more information.

Example
^^^^^^^

An example of a custom serializer and validator for a messagepart containing large binary data

.. code-block:: python

    @model_serializer()
    def _serialize_binary_file_part(self, info) -> dict[str, str]:
        """
        When we serialize, save the data as:
        - B64 of compressed data when not in zipfile mode
        - a file when in zipfile mode
        """
        if ctx := saveload.get_ctx(info):
            suffix = mimetypes.guess_extension(self.mime) or ""
            fp = ctx.save_bytes(self.as_bytes(), suffix=suffix)
            return {"_archive_path": fp, "mime": self.mime, **self._get_typekey_dict()}
        else:
            compressed_b64 = base64.b64encode(zlib.compress(self.as_bytes())).decode()
            return {"mime": self.mime, "compression": "gzip", "data": compressed_b64, **self._get_typekey_dict()}

    @model_validator(mode="wrap")
    @classmethod
    def _validate_binary_file_part(cls, v, nxt, info):
        """If the value is the URI we saved, try loading it that way."""
        assert isinstance(v, dict)
        if "_archive_path" in v:
            ctx = saveload.get_ctx(info)
            data = ctx.load_bytes(v["_archive_path"])
            return cls.from_bytes(data, mime=v["mime"])
        elif "data" in v:
            if v.get("compression") == "gzip":
                decompressed = zlib.decompress(base64.b64decode(v["data"]))
                return cls.from_bytes(mime=v["mime"], data=decompressed)
            return cls.from_b64(mime=v["mime"], data=v["data"])
        return nxt(v)
