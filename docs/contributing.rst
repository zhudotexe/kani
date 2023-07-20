Contributing
============
NLP is a fast-moving field and we couldn't dream of keeping up by ourselves. kani |:heart:| PRs!

You can contribute in many ways: helping us develop the library, citing our paper (soon!), leaving a star on GitHub
(|StarMe|_), or even just using the library!

.. |StarMe| image:: https://img.shields.io/github/stars/zhudotexe/kani?style=social&label=Star
.. _StarMe: https://github.com/zhudotexe/kani

(Funding agencies: the core devs are PhD students - you can support us that way too! |:wink:| Send us an email.)

.. todo: cite us

.. todo: examples of what is good for a PR vs what is good as a package with ext

Engines
-------
Implemented a new hot lanugage model or API? We'd love to include it in kani.

To make a PR to kani with a new engine implementation, follow these steps:

1. Add your engine implementation to ``/kani/engines``.
2. If your engine requires extra dependencies, add them as extras to ``requirements.txt`` and ``pyproject.toml``.
3. Add your engine to the docs in ``/docs/shared/engine_table.rst`` and ``/docs/engine_reference.rst``.
4. Open a PR!

Examples
--------
It's awesome to see all of the cool things people do with kani. If you've made a short, self-contained kani that
shows off some part of kani functionality that isn't covered in other examples, we'd love to see it added to our
official examples!

To make a PR with a new example, follow these steps:

1. Add your example to ``/examples/`` and document what it does in ``/examples/README.md``.
2. If your example requires extra dependencies (e.g. the ``dice`` example), make sure to mention those!
3. Open a PR!

Documentation
-------------
If you think our documentation could be improved - for example, rewording things to make it more clear,
adding diagrams, or just generally cleaning things up - open up a PR! We're happy to discuss and provide guidance with
Sphinx (our documentation generation tool).
