Customization
=============
Now that you're familiar with subclassing :class:`.Kani` in order to implement function calling, we can take a look at
the other parts you can customize.

kani is built on the philosophy that every part should be hackable. To accomplish this, kani has a set of overridable
methods you can override in a subclass. This page documents what these methods do by default, and why you might want
to override them.

.. toctree::
    :maxdepth: 2

    customization/prompting
    customization/chat_history
    customization/function_call
    customization/function_exception
