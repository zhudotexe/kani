import os
import tempfile

import pytest


@pytest.fixture
def gh_log():
    """Get the GitHub Actions step summary file."""
    fp = os.getenv("GITHUB_STEP_SUMMARY")
    if fp is None:
        with tempfile.TemporaryFile(mode="a") as f:
            yield f
    else:
        with open(fp, mode="a") as f:
            yield f
