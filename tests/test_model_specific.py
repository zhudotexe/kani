import importlib

from kani.model_specific import PARSER_REGISTRY, PROMPT_PIPELINE_REGISTRY


def test_pipeline_registry():
    for pattern, import_path in PROMPT_PIPELINE_REGISTRY:
        mod_name, attr_name = import_path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        pipe = getattr(mod, attr_name)
        assert pipe


def test_parser_registry():
    for pattern, import_path in PARSER_REGISTRY:
        mod_name, attr_name = import_path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        parser = getattr(mod, attr_name)
        assert parser
