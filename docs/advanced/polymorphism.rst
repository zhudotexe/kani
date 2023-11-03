Function "Loadouts": Polymorphism & Mixins
==========================================
What if you have some common AI functions that you want to share with multiple kani? For example, what if you've
written a solid calculator suite you want to give to two different kani - one with web retrieval and one without.

Since kani are implemented in pure Python, you can accomplish this with polymorphism!

Base Class
^^^^^^^^^^
One method is to make one of the kani a base, then subclass it to extend its functionality:

.. code-block:: python

    class BaseKani(Kani):
        @ai_function()
        def calculate(self, expr: str):
            ...

    class ChildKani(BaseKani):
        @ai_function()
        def search(self, query: str):
            ...

In this example, the ``ChildKani`` has access to both ``search`` *and* ``calculate``.

.. _mixins:

Mixins
^^^^^^
But in some cases, you won't necessarily have a single base to extend, and you might want to share multiple different
modular "loadouts" of functions. ``@ai_function``\ s don't have to be defined in :class:`.Kani` classes, so in this
case, you can use a mixin!

.. code-block:: python

    # note: the mixin isn't a subclass of Kani!
    class CalculatorMixin:
        @ai_function()
        def add(self, left: float, right: float):
            return left + right

        @ai_function()
        def mul(self, left: float, right: float):
            return left * right

    # reuse it to give a loadout of functions to multiple kani!
    class MyKani(CalculatorMixin, Kani):
        @ai_function()
        def search(self, query: str):
            ...

    class SomeOtherKani(CalculatorMixin, Kani):
        @ai_function()
        def music(self, song: str):
            ...

In this example, both the kani (``MyKani`` and ``SomeOtherKani``) have access to ``add`` and ``mul`` in addition to
the functions defined in their class body.

Just as in normal Python, you can inherit from multiple mixins. You can use this to build kani with modular sets of
functionality!
