import textwrap

PARAM_DOCS = {
    "ROLE_PARAM": (
        # ":type role: ChatRole | list[ChatRole]\n"
        ":param role: The role (if a single role is given) or roles (if a list is given) to apply this operation to. If"
        " not set, ignores the role of the message."
    ),
    "PREDICATE_PARAM": (
        # ":type predicate: Callable[[ChatMessage], bool]\n"
        ":param predicate: A function that takes a :class:`.ChatMessage` and returns a boolean specifying whether to"
        " operate on this message or not."
    ),
    "MULTIFILTER": (
        "If multiple filter params are supplied, this method will only operate on messages that match ALL of the"
        " filters."
    ),
}

PARAM_DOCS["ALL_FILTERS"] = "\n\n".join(PARAM_DOCS.values())


def autoparams(f):
    """A decorator to template in the RST-formatted param docs above into docstrings."""
    f.__doc__ = textwrap.dedent(f.__doc__).format(**PARAM_DOCS)
    return f
