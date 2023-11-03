Build the Chat Prompt
=====================
Modern language models work by generating a *continuation* to a *prompt*. The prompt contains all the context-specific
knowledge the model has access to; if it's not in the prompt, it can only use its pretraining data.

While we would love to pass the entire chat history in the prompt, models also contain a *token limit* - the maximum
size of the prompt we can give it at one time.

.. hint:: "token limit" is also known as "context size".

Since chats can be longer than a model's token limit, we have to decide which parts to keep and which parts to omit,
creating a sliding window of memory for the LM.

.. important::

    Language models can't remember what happened in a conversation beyond their token limit. Making them do so is a hot
    area of research!

By default, kani includes the **system prompt** and any messages specified as **always include** (in the initializer),
then as many messages as possible fit in the remaining token limit, prioritizing later messages.

.. todo: figure demonstrating this

To override this behaviour, override :meth:`.Kani.get_prompt` in your subclass:

.. automethod:: kani.Kani.get_prompt
    :noindex:

For example, here's how you might override the behaviour to only include the most recent 4 messages
(omitting earlier ones to fit in the token length if necessary) and any always included messages:

.. seealso::

    This example is available in the
    `GitHub repo <https://github.com/zhudotexe/kani/blob/main/examples/3_customization_last_four.py>`__.

.. code-block:: python

    class LastFourKani(Kani):
        async def get_prompt(self):
            # calculate how many tokens we have for the prompt, accounting for the system prompt,
            # always_included_messages, any tokens reserved by the engine, and the response
            remaining = self.max_context_size - self.always_len
            # working backwards through history...
            messages = []
            for message in reversed(self.chat_history[-4:]):
                # if the message fits in the space we have remaining...
                message_len = self.message_token_len(message)
                remaining -= message_len
                if remaining > 0:
                    # add it to the returned prompt!
                    messages.insert(0, message)
                else:
                    break
            return self.always_included_messages + messages

Chatting with this kani, we can see that it loses any memory of what happened more than 4 messages (2 rounds) ago:

.. code-block:: pycon

    >>> chat_in_terminal(LastFourKani(engine))
    USER: Hi kani! My name is Andrew.
    AI: Hello Andrew! How can I assist you today?

    USER: What does "kani" mean in Japanese?
    AI: "Kani" in Japanese means "Crab".

    USER: How do you pronounce it?
    AI: Kani is pronounced as "kah-nee" in Japanese.

    USER: What is my name?
    AI: As an AI, I don't have access to personal data about individuals unless it has
    been shared with me in the course of our conversation. I'm designed to respect user
    privacy and confidentiality.
