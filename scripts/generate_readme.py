from pathlib import Path


def load_md(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return text.rstrip() + "\n\n"


header = """
<p align="center">
  <img src="assets/logo.png" alt="PopGames logo" width="300">
</p>


A Python package to model and simulate population games.

[![PyPI version](https://img.shields.io/pypi/v/popgames.svg)](https://pypi.org/project/popgames/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/popgames.svg)](https://pypi.org/project/popgames/)
[![docs](https://img.shields.io/badge/docs-online-success)](https://martinez-piazuelo.github.io/popgames/)

---

## Documentation

Full API reference and usage examples are available at:

[https://martinez-piazuelo.github.io/popgames/](https://martinez-piazuelo.github.io/popgames/)

---

"""

docs = [
    load_md("docs/source/getting_started/installation.md"),
    load_md("docs/source/getting_started/quick_example.md").replace("$", "$$"),
    load_md("docs/source/contributing/index.md").replace("#", "##"),
]

readme = header
for doc in docs:
    readme += doc

Path("README.md").write_text(readme.rstrip() + "\n", encoding="utf-8")
