import typing
from typing import Annotated


# ==== AIParam ====
class AIParam:
    def __init__(self, desc: str):
        self.desc = desc


def get_aiparam(annotation: type) -> AIParam | None:
    """If the type annotation is an Annotated containing an AIParam, extract and return it."""
    # is it Annotated? if so, get AIParam from annotation
    if typing.get_origin(annotation) is not Annotated:
        return

    for a in typing.get_args(annotation):
        if isinstance(a, AIParam):
            return a
