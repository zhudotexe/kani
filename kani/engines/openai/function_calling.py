"""
As OpenAI doesn't tell us exactly how functions are exposed to GPT, we have to rely on some community reverse
engineering to build a reliable method to reserve tokens for AI Functions.

See https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573 for more details.
"""

import json
import warnings

from kani.ai_function import AIFunction


def prompt(functions: list[AIFunction]) -> str:
    return "".join(map(format_function, functions))


def format_function(function: AIFunction) -> str:
    # Thanks @CGamesPlay for https://gist.github.com/CGamesPlay/dd4f108f27e2eec145eedf5c717318f5, which this
    # implementation is based on.
    def resolve_ref(schema):
        if schema.get("$ref") is not None:
            *_, ref = schema["$ref"].rsplit("/", 1)
            schema = json_schema["$defs"][ref]
        return schema

    def format_schema(schema, indent):
        schema = resolve_ref(schema)
        if "enum" in schema:
            return format_enum(schema)
        elif schema["type"] == "object":
            return format_object(schema, indent)
        elif schema["type"] == "array":
            return format_schema(schema["items"], indent) + "[]"
        elif schema["type"] in ("string", "number", "integer", "boolean", "null"):  # these are all 1 token!
            return schema["type"]
        warnings.warn(
            f"Unknown JSON schema type estimating tokens for OpenAI: {schema['type']!r}\n"
            "The returned estimate may be off by a significant amount."
        )
        return schema["type"]

    def format_enum(schema):
        return " | ".join(json.dumps(o) for o in schema["enum"])

    def format_object(schema, indent):
        result = "{\n"
        if "properties" not in schema or len(schema["properties"]) == 0:
            if schema.get("additionalProperties", False):
                return "object"
            return None
        for key, value in schema["properties"].items():
            value = resolve_ref(value)
            value_rendered = format_schema(value, indent + 1)
            if value_rendered is None:
                continue
            if "description" in value and indent == 0:
                for line in value["description"].strip().split("\n"):
                    result += f"{'  ' * indent}// {line}\n"
            optional = "" if key in schema.get("required", {}) else "?"
            comment = "" if value.get("default") is None else f" // default: {format_default(value)}"
            result += f"{'  ' * indent}{key}{optional}: {value_rendered},{comment}\n"
        result += ("  " * (indent - 1)) + "}"
        return result

    def format_default(schema):
        v = schema["default"]
        if schema["type"] == "number":
            return f"{v:.1f}" if float(v).is_integer() else str(v)
        else:
            return str(v)

    json_schema = function.json_schema
    if function.desc:
        out = f"// {function.desc}\ntype {function.name} = ("
    else:
        out = f"type {function.name} = ("
    formatted = format_object(json_schema, 0)
    if formatted is not None:
        out += "_: " + formatted
    out += ") => any;\n\n"
    return out
