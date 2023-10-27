Installation
============

TL;DR: kani requires Python 3.10+. You can install kani though pip:

.. code-block:: console

    $ pip install kani

Engines
-------
Since kani is designed to be model-agnostic, kani uses :doc:`engines` to interface with different language model
backends. These engines usually have their own dependencies beyond kani's, which can be quite large (e.g.
``transformers``).

To install these dependencies along with kani, you should specify one or more extras with your installation command
(e.g. ``pip install "kani[openai]"``). The table below lists the engines that are included with kani and the extra you
should use to install its dependencies.

.. include:: shared/engine_table.rst

.. seealso::
    Extensions provide additional engines you can use to extend kani with additional functionality like support for
    vision-language models and ReAct prompting. See :doc:`community/extensions` for a list of available extensions!

.. seealso::
    Want to use a different model with kani? Check out :doc:`engines` for details on how to implement the common
    Engine interface and use kani with any LM.

Installing on Conda
-------------------
You may need to install pip in your conda environment first:

.. code-block:: console

    $ conda install pip

Then, follow the instructions for installing with pip above.

.. caution::

    In certain cases when using a conda venv, the ``pip`` binary may
    `reference a different environment <https://stackoverflow.com/questions/41060382/using-pip-to-install-packages-to-anaconda-environment>`_,
    and kani may appear to be uninstalled even if pip assures that it is. Using ``python -m pip install`` in place of
    ``pip install`` may mitigate this issue.

Virtual Environment
-------------------
If you're not using conda, we recommend using a virtual environment to manage your project dependencies. This will
help you prevent polluting your global Python installation with all sorts of packages.

.. tab:: macOS/Linux

    .. code-block:: console

        $ python -m venv ./venv
        $ ./venv/bin/activate
        $ pip install "kani[...]"

.. tab:: Windows

    .. code-block:: console

        $ python -m venv venv
        $ venv\Scripts\activate.bat
        $ pip install "kani[...]"

Development Version
-------------------
If you'd like to install the development version of kani, you can install it from GitHub directly:

.. code-block:: console

    $ pip install 'kani @ git+https://github.com/zhudotexe/kani.git@main'

This will install the latest version of kani.

.. note::
    You may need to use ``pip install --upgrade --no-deps --force-reinstall ...`` to force pip to re-fetch the
    latest kani from GitHub.

    To install an engine's extras, use ``pip install 'kani[...] @ git+https://github.com/zhudotexe/kani.git@main'``.

.. caution::
    Development versions of kani may be unstable! Do not use development kani in production or in final research
    experiments; pin a released version of kani instead.

Requirements File
-----------------
If you're running experiments using kani, we recommend pinning the version of kani to ensure your runs are reproducible.
To do this, we recommend storing all your Python requirements in a ``requirements.txt`` file.

.. code-block:: text

    kani[...]==x.y.z
    # ... other dependencies

You can automatically generate this file too, by running ``pip freeze > requirements.txt``.

Later, anyone else running your code can install the same dependency versions by simply running
``pip install -r requirements.txt``.

Next, we'll take a look at basic usage of kani.