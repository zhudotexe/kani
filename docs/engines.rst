Engines
=======
the builtin ones and how to build them

Each engine must implement two methods: :meth:`.BaseEngine.message_len`, which takes a single :class:`.ChatMessage` and
returns the length of that message, in tokens, and :meth:`.BaseEngine.predict`, which is responsible for taking
a list of :class:`.ChatMessage` and :class:`.AIFunction` (discussed in the next section) and returning a new
:class:`.BaseCompletion`.