"""
We use the render_tool_namespace function from the GPT-OSS chat template, since the format hasn't seemed to change
since GPT-3.5.

Chat template: https://huggingface.co/openai/gpt-oss-20b?chat_template=default

See https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573 for more details.
"""

import json


def prompt(functions: list[dict]) -> str:
    return "# Tools\n\n" + render_tool_namespace("functions", functions)


def render_typescript_type(param_spec, required_params):
    out = []

    # {%- if param_spec.type == "array" -%}
    #     {%- if param_spec['items'] -%}
    #         {%- if param_spec['items']['type'] == "string" -%}
    #             {{- "string[]" }}
    #         {%- elif param_spec['items']['type'] == "number" -%}
    #             {{- "number[]" }}
    #         {%- elif param_spec['items']['type'] == "integer" -%}
    #             {{- "number[]" }}
    #         {%- elif param_spec['items']['type'] == "boolean" -%}
    #             {{- "boolean[]" }}
    #         {%- else -%}
    #             {%- set inner_type = render_typescript_type(param_spec['items'], required_params) -%}
    #             {%- if inner_type == "object | object" or inner_type|length > 50 -%}
    #                 {{- "any[]" }}
    #             {%- else -%}
    #                 {{- inner_type + "[]" }}
    #             {%- endif -%}
    #         {%- endif -%}
    #         {%- if param_spec.nullable -%}
    #             {{- " | null" }}
    #         {%- endif -%}
    #     {%- else -%}
    #         {{- "any[]" }}
    #         {%- if param_spec.nullable -%}
    #             {{- " | null" }}
    #         {%- endif -%}
    #     {%- endif -%}
    if param_spec.get("type") == "array":
        if param_spec["items"]:
            if param_spec["items"].get("type") == "string":
                out.append("string[]")
            elif param_spec["items"].get("type") == "number":
                out.append("number[]")
            elif param_spec["items"].get("type") == "integer":
                out.append("number[]")
            elif param_spec["items"].get("type") == "boolean":
                out.append("boolean[]")
            else:
                inner_type = render_typescript_type(param_spec["items"], required_params)
                if inner_type == "object | object" or len(inner_type) > 50:
                    out.append("any[]")
                else:
                    out.append(f"{inner_type}[]")
        else:
            out.append("any[]")

        if param_spec.get("nullable"):
            out.append(" | null")

    # {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}
    #     {#- Handle array of types like ["object", "object"] from Union[dict, list] #}
    #     {%- if param_spec.type | length > 1 -%}
    #         {{- param_spec.type | join(" | ") }}
    #     {%- else -%}
    #         {{- param_spec.type[0] }}
    #     {%- endif -%}
    elif "type" in param_spec and isinstance(param_spec["type"], list) and len(param_spec["type"]) > 0:
        # Handle array of types like ["object", "object"] from Union[dict, list]
        if len(param_spec["type"]) > 1:
            out.append(" | ".join(param_spec["type"]))
        else:
            out.append(param_spec["type"][0])
    # {%- elif param_spec.oneOf -%}
    #     {#- Handle oneOf schemas - check for complex unions and fallback to any #}
    #     {%- set has_object_variants = false -%}
    #     {%- for variant in param_spec.oneOf -%}
    #         {%- if variant.type == "object" -%}
    #             {%- set has_object_variants = true -%}
    #         {%- endif -%}
    #     {%- endfor -%}
    #     {%- if has_object_variants and param_spec.oneOf|length > 1 -%}
    #         {{- "any" }}
    #     {%- else -%}
    #         {%- for variant in param_spec.oneOf -%}
    #             {{- render_typescript_type(variant, required_params) -}}
    #             {%- if variant.description %}
    #                 {{- "// " + variant.description }}
    #             {%- endif -%}
    #             {%- if variant.default is defined %}
    #                 {{ "// default: " + variant.default|tojson }}
    #             {%- endif -%}
    #             {%- if not loop.last %}
    #                 {{- " | " }}
    #             {% endif -%}
    #         {%- endfor -%}
    #     {%- endif -%}
    elif param_spec.get("oneOf"):
        # Handle oneOf schemas - check for complex unions and fallback to any
        has_object_variants = False
        for variant in param_spec["oneOf"]:
            if variant["type"] == "object":
                has_object_variants = True
        if has_object_variants and len(param_spec["oneOf"]) > 1:
            out.append("any")
        else:
            variants = []
            for variant in param_spec["oneOf"]:
                variants.append(render_typescript_type(variant, required_params))
                if variant.get("description"):
                    variants.append(f"// {variant['description']}")
                if "default" in variant:
                    variants.append(f"// default: {json.dumps(variant['default'])}")
            out.append(" | ".join(variants))

    # {%- elif param_spec.type == "string" -%}
    #     {%- if param_spec.enum -%}
    #         {{- '"' + param_spec.enum|join('" | "') + '"' -}}
    #     {%- else -%}
    #         {{- "string" }}
    #         {%- if param_spec.nullable %}
    #             {{- " | null" }}
    #         {%- endif -%}
    #     {%- endif -%}
    elif param_spec.get("type") == "string":
        if param_spec.get("enum"):
            out.append('"' + '" | "'.join(param_spec["enum"]) + '"')
        else:
            out.append("string")
            if param_spec.get("nullable"):
                out.append(" | null")
    # {%- elif param_spec.type == "number" -%}
    #     {{- "number" }}
    elif param_spec.get("type") == "number":
        out.append("number")
    # {%- elif param_spec.type == "integer" -%}
    #     {{- "number" }}
    elif param_spec.get("type") == "integer":
        out.append("number")
    # {%- elif param_spec.type == "boolean" -%}
    #     {{- "boolean" }}
    elif param_spec.get("type") == "boolean":
        out.append("boolean")
    # {%- elif param_spec.type == "object" -%}
    #     {%- if param_spec.properties -%}
    #         {{- "{\n" }}
    #         {%- for prop_name, prop_spec in param_spec.properties.items() -%}
    #             {{- prop_name -}}
    #             {%- if prop_name not in (param_spec.required or []) -%}
    #                 {{- "?" }}
    #             {%- endif -%}
    #             {{- ": " }}
    #             {{ render_typescript_type(prop_spec, param_spec.required or []) }}
    #             {%- if not loop.last -%}
    #                 {{-", " }}
    #             {%- endif -%}
    #         {%- endfor -%}
    #         {{- "}" }}
    #     {%- else -%}
    #         {{- "object" }}
    #     {%- endif -%}
    elif param_spec.get("type") == "object":
        if param_spec.get("properties"):
            out.append("{\n")
            props = []
            for prop_name, prop_spec in param_spec["properties"].items():
                prop = prop_name
                if prop_name not in (param_spec.get("required") or []):
                    prop += "?"
                prop += ": "
                prop += render_typescript_type(prop_spec, param_spec.get("required") or [])
                props.append(prop)
            out.append(", ".join(props))
            out.append("}")
        else:
            out.append("object")
    # {%- else -%}
    #     {{- "any" }}
    else:
        out.append("any")

    return "".join(out)


def render_tool_namespace(namespace_name, tools):
    # {{- "## " + namespace_name + "\n\n" }}
    # {{- "namespace " + namespace_name + " {\n\n" }}
    out = [f"## {namespace_name}\n\nnamespace {namespace_name} {{\n\n"]
    # {%- for tool in tools %}
    #     {%- set tool = tool.function %}
    #     {{- "// " + tool.description + "\n" }}
    #     {{- "type "+ tool.name + " = " }}
    for tool in tools:
        tool = tool["function"]
        out.append(f"// {tool.get('description', '')}\n")
        out.append(f"type {tool['name']} = ")
        # {%- if tool.parameters and tool.parameters.properties %}
        #     {{- "(_: {\n" }}
        #     {%- for param_name, param_spec in tool.parameters.properties.items() %}
        #         {%- if param_spec.description %}
        #             {{- "// " + param_spec.description + "\n" }}
        #         {%- endif %}
        #         {{- param_name }}
        #         {%- if param_name not in (tool.parameters.required or []) -%}
        #             {{- "?" }}
        #         {%- endif -%}
        #         {{- ": " }}
        #         {{- render_typescript_type(param_spec, tool.parameters.required or []) }}
        #         {%- if param_spec.default is defined -%}
        #             {%- if param_spec.enum %}
        #                 {{- ", // default: " + param_spec.default }}
        #             {%- elif param_spec.oneOf %}
        #                 {{- "// default: " + param_spec.default }}
        #             {%- else %}
        #                 {{- ", // default: " + param_spec.default|tojson }}
        #             {%- endif -%}
        #         {%- endif -%}
        #         {%- if not loop.last %}
        #             {{- ",\n" }}
        #         {%- else %}
        #             {{- ",\n" }}
        #         {%- endif -%}
        #     {%- endfor %}
        #     {{- "}) => any;\n\n" }}
        if tool.get("parameters") and tool["parameters"].get("properties"):
            out.append("(_: {\n")
            for param_name, param_spec in tool["parameters"]["properties"].items():
                if param_spec.get("description"):
                    out.append(f"// {param_spec['description']}\n")
                out.append(param_name)
                if param_name not in (tool["parameters"].get("required") or []):
                    out.append("?")
                out.append(": ")
                out.append(render_typescript_type(param_spec, tool["parameters"].get("required") or []))
                if "default" in param_spec:
                    if param_spec.get("enum"):
                        out.append(f", // default: {param_spec['default']}")
                    elif param_spec.get("oneOf"):
                        out.append(f"// default: {param_spec['default']}")
                    else:
                        out.append(f", // default: {json.dumps(param_spec['default'])}")
                out.append(",\n")
            out.append("}) => any;\n\n")
        # {%- else -%}
        #     {{- "() => any;\n\n" }}
        # {%- endif -%}
        else:
            out.append("() => any;\n\n")
    # {{- "} // namespace " + namespace_name }}
    out.append(f"}} // namespace {namespace_name}")

    return "".join(out)
