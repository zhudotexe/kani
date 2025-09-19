Engines
=======
Engines are the means by which kani interact with language models. As you've seen, kani comes with a few engines
included:

.. include:: shared/engine_table.rst

In this section, we'll cover engine-specific details and features you might want to be aware of when using certain
engines.

We'll also discuss how to implement your own engine to use any language model or API you can think of.

.. tip::

    Built an engine for a model kani doesn't support yet?
    kani is OSS and |:heart:| PRs with engine implementations for the latest models - see :doc:`community/contributing`.

.. toctree::
    :maxdepth: 2

    engines/implementing
    engines/anthropic
    engines/google
    engines/huggingface
    engines/llamacpp
    engines/openai
