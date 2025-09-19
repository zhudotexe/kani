"""Helper utils for importing optional extension pkgs."""


class _NotInstalledHelper:
    def __init__(self, pkg: str):
        self.pkg = pkg

    def __getattr__(self, _):
        raise ImportError(
            f'This method requires an additional package to be installed. Use `pip install "{self.pkg}"` to install'
            " additional dependencies."
        )


# kani-multimodal-core
try:
    import kani.ext.multimodal_core as multimodal_core
    from kani.ext.multimodal_core import cli as multimodal_cli

    has_multimodal_core = True
except ImportError:
    multimodal_core = _NotInstalledHelper("kani-multimodal-core")
    multimodal_cli = _NotInstalledHelper("kani-multimodal-core")
    has_multimodal_core = False
