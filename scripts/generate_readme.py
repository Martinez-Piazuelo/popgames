import re
from pathlib import Path

README_REPLACE_PATTERN = re.compile(
    r"""
    (?P<original>
        (?:
            ```math.*?```      |   # fenced math block
            \$\$.*?\$\$            # $$ math block
        )
    )
    \s*
    <!--\s*README_REPLACE_START\s*-->
    \s*
    (?P<replacement>.*?)
    \s*
    <!--\s*README_REPLACE_END\s*-->
    """,
    re.DOTALL | re.VERBOSE,
)


def load_md(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return text.rstrip() + "\n\n"


def demote_headings(text: str, levels: int = 1) -> str:
    """Demote Markdown headings by the given number of levels."""

    def repl(match: re.Match[str]) -> str:
        hashes = match.group(1)
        new_level = min(len(hashes) + levels, 6)
        return "#" * new_level + " "

    return re.sub(r"^(#{1,6})\s+", repl, text, flags=re.MULTILINE)


def apply_readme_overrides(text: str) -> str:
    """
    Replace a docs math block with the README-specific replacement that follows it.

    Expected pattern in the docs:

    $$ ... $$
    <!-- README_REPLACE_START -->
    ```math
    ...
    ```
    <!-- README_REPLACE_END -->
    """

    def repl(match: re.Match[str]) -> str:
        replacement = match.group("replacement").strip()
        return replacement + "\n"

    return README_REPLACE_PATTERN.sub(repl, text)


def remove_readme_comments(text: str) -> str:
    """Remove any leftover README replacement comments if present."""
    text = re.sub(r"<!--\s*README_REPLACE_START\s*-->", "", text)
    text = re.sub(r"<!--\s*README_REPLACE_END\s*-->", "", text)
    return text


def normalize_newlines(text: str) -> str:
    """Collapse excessive blank lines while keeping Markdown readable."""
    return re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"


header = (
    """
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
""".strip()
    + "\n\n"
)


docs = [
    load_md("docs/source/getting_started/installation.md"),
    load_md("docs/source/getting_started/quick_example.md"),
    demote_headings(load_md("docs/source/contributing/index.md"), levels=1),
]

readme = header + "".join(docs)
readme = apply_readme_overrides(readme)
readme = remove_readme_comments(readme)
readme = normalize_newlines(readme)

Path("README.md").write_text(readme, encoding="utf-8")
