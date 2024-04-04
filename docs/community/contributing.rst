Contributing
============
NLP is a fast-moving field and we couldn't dream of keeping up by ourselves. kani |:heart:| PRs!

You can contribute in many ways: helping us develop the library,
`citing our paper <https://aclanthology.org/2023.nlposs-1.8/>`_, leaving a star on GitHub
(|StarMe|_), or even just using the library!

.. |StarMe| image:: https://img.shields.io/github/stars/zhudotexe/kani?style=social&label=Star
.. _StarMe: https://github.com/zhudotexe/kani

(Funding agencies: the core devs are PhD students - you can support us that way too! |:wink:| Send us an email.)

Contributing Code
-----------------
When contributing code and opening PRs, there are a couple things to keep in mind. kani is built on the principles
of being lightweight and unopinionated in the core library - which means we may suggest that certain contributions
are better suited as an example or a 3rd-party extension package (see :doc:`extensions`) rather than an
addition to the core library.

**Helpful PRs**

- New generic engines and a concrete implementation
- Engines for pretrained base models (e.g. not a fine-tune)
- Examples of functionality that isn't demonstrated in existing examples
- Illustrative figures and images

**Better as 3rd Party**

- Specific prompt frameworks
- Cool new use cases whose core idea is already covered by examples

.. note::
    For 3rd party packages, we recommend using the ``kani.ext.*`` namespace (e.g. ``kani.ext.my_cool_extension``).

    To get started with building a kani extension package, see :doc:`extensions`!

If you're unsure whether a contribution is better suited as a core library contribution or an extension package,
open up a discussion on the GitHub repository.
We're also happy to showcase cool projects using kani (see :doc:`showcase`)!

That said, if you think you have an idea that fits well into the kani core library, read on!

Engines
-------
Implemented a new hot language model or API? We'd love to include it in kani.

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
