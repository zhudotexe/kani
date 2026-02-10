"""
Clean up any _request.http and <hash>.prompt.txt without a corresponding response.
"""

from pathlib import Path

cache_root = Path(__file__).parent

# api
for fp in cache_root.glob("*/*/*/_request.http"):
    if len(list(fp.parent.iterdir())) == 1:
        fp.unlink()
        fp.parent.rmdir()
        print(fp.parent)

# local
for fp in cache_root.glob("*/*/*/prompt.txt"):
    if len(list(fp.parent.iterdir())) == 1:
        fp.unlink()
        fp.parent.rmdir()
        print(fp.parent)

# clean up empty dirs
for fp in cache_root.glob("*/*/*"):
    if fp.is_dir() and len(list(fp.iterdir())) == 0:
        fp.rmdir()
        print(fp)
