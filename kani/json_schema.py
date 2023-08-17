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

    def __init__(self, name: str, t: type, default, aiparam: Optional["AIParam"] = None):
        self.name = name
        self.type = t
        self.default = default
        self.aiparam = aiparam

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


class JSONSchemaBuilder(pydantic.json_schema.GenerateJsonSchema):
    """Subclass of the Pydantic JSON schema builder to provide more fine-grained control over titles and refs."""

    def field_title_should_be_set(self, _) -> bool:
        # We never want titles to be set.
        return False

    def _update_class_schema(self, json_schema, title, *args, **kwargs):
        # don't set title, transform it to None
        return super()._update_class_schema(json_schema, None, *args, **kwargs)

    def _build_definitions_remapping(self):
        # we need to remember what remappings Pydantic did so we can flatten them later
        self.remapping = super()._build_definitions_remapping()
        return self.remapping

    def flatten_singleton_refs(self, json_schema, threshold=1):
        """Substitute any refs that only occur once with the literal."""
        defs = json_schema.get("$defs")
        if defs is None:
            return json_schema
        def_counts = self.get_json_ref_counts(json_schema)

        def _flatten(obj):
            if not isinstance(obj, dict):
                return obj

            new_obj = {}
            for k, v in obj.items():
                if isinstance(v, list):
                    new_obj[k] = [_flatten(x) for x in v]
                elif isinstance(v, dict):
                    new_obj[k] = _flatten(v)
                elif k == "$ref" and def_counts[v] <= threshold:
                    new_obj.update(self.get_schema_from_definitions(v))
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

    def flatten_singleton_allof(self, json_schema, flatten_refs=False):
        """Bring any allOf that only have a single child up to the sibling level."""

        def _flatten(obj):
            if not isinstance(obj, dict):
                return obj

            for k, v in obj.items():
                if isinstance(v, list):
                    obj[k] = [_flatten(x) for x in v]
                elif isinstance(v, dict):
                    if "allOf" in v and len(v["allOf"]) == 1:
                        entry = v["allOf"][0]
                        if "$ref" not in entry or flatten_refs:
                            v.pop("allOf")
                            v.update(entry)
                    obj[k] = _flatten(v)
            return obj

        return _flatten(json_schema)

    def remove_titles(self, json_schema):
        """Recursively remove title tags from the schema in place."""
        if not isinstance(json_schema, dict):
            return
        # only kill the title key if it's a str; a prop could be named "title"
        if "title" in json_schema and isinstance(json_schema["title"], str):
            json_schema.pop("title")
        # then recurse
        for k, v in json_schema.items():
            if isinstance(v, list):
                for x in v:
                    self.remove_titles(x)
            elif isinstance(v, dict):
                self.remove_titles(v)

    def generate(self, *args, **kwargs):
        json_schema = super().generate(*args, **kwargs)
        # take the remappings and make it canonical
        new_json_to_defs_refs = {
            new: self.remapping.defs_remapping[self.json_to_defs_refs[old]]
            for old, new in self.remapping.json_remapping.items()
        }
        self.json_to_defs_refs = new_json_to_defs_refs
        new_definitions = {new: self.definitions[old] for old, new in self.remapping.defs_remapping.items()}
        self.definitions = new_definitions
        # flatten any singleton def/refs
        json_schema = self.flatten_singleton_refs(json_schema, 2)
        # flatten singleton allOf
        json_schema = self.flatten_singleton_allof(json_schema)
        # recursively remove title tags
        self.remove_titles(json_schema)
        return json_schema


def create_json_schema(params: list[AIParamSchema]) -> dict:
    """Create a JSON schema from a list of parameters to an AIFunction.

    There are some subtle differences compared to how Pydantic creates JSON schemas by default; most notably,
    some titles are omitted to minimize the token count, and defaults are not exposed.
    """
    # create pydantic fields for each AIParam
    fields = {}
    for param in params:
        field_kwargs = dict(description=param.description)
        if not param.required:
            field_kwargs["default"] = param.default
        fields[param.name] = (param.type, pydantic.Field(**field_kwargs))
    # create a temp model for generating json schemas
    pydantic_model = pydantic.create_model("_FunctionSpec", **fields)
    return pydantic_model.model_json_schema(schema_generator=JSONSchemaBuilder, ref_template=REF_TEMPLATE)
