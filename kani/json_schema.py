import collections
import inspect
import typing
from typing import TYPE_CHECKING, Optional

import pydantic

if TYPE_CHECKING:
    from .ai_function import AIParam

# this is the same as Pydantic's as of v2.1, but we specify it here because some downstream things rely on it
# (e.g. engines.openai.function_calling_tokens)
REF_TEMPLATE = "#/$defs/{model}"


class AIParamSchema:
    """Used to annotate parameters of AIFunctions in order to make generating their schema nicer.

    This class is only used internally within kani and generally shouldn't be constructed manually.
    """

    def __init__(self, name: str, t: type, default, aiparam: Optional["AIParam"], inspect_param: inspect.Parameter):
        self.name = name
        self.type = t  # will not include Annotated if present
        self.default = default
        self.aiparam = aiparam
        self.inspect_param = inspect_param

    @property
    def required(self):
        return self.default is inspect.Parameter.empty

    @property
    def origin_type(self):
        """If the type takes parameters (e.g. list[...]), the base type (i.e. list). Otherwise same as the type."""
        return typing.get_origin(self.type) or self.type

    @property
    def description(self):
        return self.aiparam.desc if self.aiparam is not None else None

    @property
    def title(self):
        return self.aiparam.title if self.aiparam is not None else None

    def __str__(self):
        default = ""
        if not self.required:
            default = f" = {self.default!r}"
        annotation = inspect.formatannotation(self.type)
        return f"{self.name}: {annotation}{default}"


class JSONSchemaBuilder(pydantic.json_schema.GenerateJsonSchema):
    """Subclass of the Pydantic JSON schema builder to provide more fine-grained control over titles and refs."""

    def field_title_should_be_set(self, schema) -> bool:
        # We only want titles to be set if the field explicitly set it
        return super().field_title_should_be_set(schema) and schema.get("title") is not None

    def flatten_singleton_refs(self, json_schema, threshold=1):
        """Substitute any refs that only occur once with the literal."""
        defs = json_schema.get("$defs")
        if defs is None:
            return json_schema

        def_counts = collections.Counter()

        def _count_refs(obj):
            for k, v in obj.items():
                if isinstance(v, list):
                    [_count_refs(x) for x in v if isinstance(x, dict)]
                elif isinstance(v, dict):
                    _count_refs(v)
                elif k == "$ref":
                    def_counts[v] += 1

        _count_refs(json_schema)

        def _flatten(obj):
            if not isinstance(obj, dict):
                return obj

            new_obj = {}
            for k, v in obj.items():
                if isinstance(v, list):
                    new_obj[k] = [_flatten(x) for x in v]
                elif isinstance(v, dict):
                    new_obj[k] = _flatten(v)
                elif k == "$ref":
                    # FIXME this is hardcoded for now since the def template isn't likely to change, but a bit fragile
                    def_name = v.removeprefix("#/$defs/")
                    if def_counts[def_name] <= threshold:
                        new_obj.update(defs[def_name])
                    else:
                        new_obj[k] = v
                else:
                    new_obj[k] = v
            return new_obj

        json_schema = _flatten(json_schema)

        self._garbage_collect_definitions(json_schema)
        if self.definitions:
            json_schema["$defs"] = self.definitions
        else:
            json_schema.pop("$defs")
        return json_schema

    def generate(self, *args, **kwargs):
        json_schema = super().generate(*args, **kwargs)
        # flatten any singleton def/refs
        json_schema = self.flatten_singleton_refs(json_schema, 2)
        return json_schema


def create_json_schema(params: list[AIParamSchema], name: str = "_FunctionSpec", desc: str = None) -> dict:
    """Create a JSON schema from a list of parameters to an AIFunction.

    There are some subtle differences compared to how Pydantic creates JSON schemas by default; most notably:
    - singleton refs to sub-models are inserted in-place rather than requiring a ref to another key
    - the titles of parameters are omitted unless an AIParam explicitly sets its ``title``
    """
    # create pydantic fields for each AIParam
    fields = {}
    for param in params:
        field_kwargs = dict(description=param.description, title=param.title)
        if not param.required:
            field_kwargs["default"] = param.default
        fields[param.name] = (param.type, pydantic.Field(**field_kwargs))
    # create a temp model for generating json schemas
    pydantic_model = pydantic.create_model(name, __doc__=desc, **fields)
    return pydantic_model.model_json_schema(schema_generator=JSONSchemaBuilder, ref_template=REF_TEMPLATE)
