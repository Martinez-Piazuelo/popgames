import re
from pathlib import Path


def load_md(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return text.rstrip() + "\n\n"


def split_code_fences(text: str) -> list[tuple[str, str]]:
    """Split markdown into ('text'|'code', content) chunks."""
    parts: list[tuple[str, str]] = []
    pattern = re.compile(r"```.*?```", re.DOTALL)
    last = 0

    for match in pattern.finditer(text):
        if match.start() > last:
            parts.append(("text", text[last : match.start()]))
        parts.append(("code", match.group(0)))
        last = match.end()

    if last < len(text):
        parts.append(("text", text[last:]))

    return parts


def normalize_math_expr(expr: str) -> str:
    """Make math more GitHub-friendly."""
    expr = expr.strip()

    expr = expr.replace(r"\begin{bmatrix}", r"\begin{pmatrix}")
    expr = expr.replace(r"\end{bmatrix}", r"\end{pmatrix}")

    return expr


def convert_inline_math_to_math_block(text: str) -> str:
    """
    Convert inline math $...$ into fenced ```math blocks.
    This is more reliable on GitHub than trying to keep complex inline math.
    """
    pattern = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        expr = normalize_math_expr(match.group(1))
        return f"\n```math\n{expr}\n```\n"

    return pattern.sub(repl, text)


def convert_display_math_to_math_block(text: str) -> str:
    """
    Convert $$...$$ blocks into fenced ```math blocks.
    """
    pattern = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        expr = normalize_math_expr(match.group(1))
        return f"\n```math\n{expr.strip()}\n```\n"

    return pattern.sub(repl, text)


def simplify_problematic_math(text: str) -> str:
    """
    README-only simplifications for expressions that often render poorly on GitHub.
    """
    text = text.replace(
        r"\mathbf{f}(\mathbf{x}) = \begin{pmatrix} R & S \\ T & P \end{pmatrix}\mathbf{x}",
        r"\mathbf{f}(\mathbf{x}) = A\mathbf{x}, \quad A = \begin{pmatrix} R & S \\ T & P \end{pmatrix}",
    )
    text = text.replace(
        r"\mathbf{f}(\mathbf{x})=\begin{pmatrix}R&S\\T&P\end{pmatrix}\mathbf{x}",
        r"\mathbf{f}(\mathbf{x}) = A\mathbf{x}, \quad A = \begin{pmatrix} R & S \\ T & P \end{pmatrix}",
    )
    return text


def convert_markdown_math_for_github(text: str) -> str:
    """
    Convert markdown math conservatively for GitHub README rendering.
    Does not touch non-math fenced code blocks.
    """
    chunks = split_code_fences(text)
    out: list[str] = []

    for kind, chunk in chunks:
        if kind == "code":
            out.append(chunk)
        else:
            chunk = simplify_problematic_math(chunk)
            chunk = convert_display_math_to_math_block(chunk)
            chunk = convert_inline_math_to_math_block(chunk)
            chunk = re.sub(r"\n{3,}", "\n\n", chunk)
            out.append(chunk)

    return "".join(out)


def demote_headings(text: str, levels: int = 1) -> str:
    """Demote Markdown headings by the given number of levels."""

    def repl(match: re.Match[str]) -> str:
        hashes = match.group(1)
        return "#" * (len(hashes) + levels) + " "

    return re.sub(r"^(#{1,6})\s+", repl, text, flags=re.MULTILINE)


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
readme = convert_markdown_math_for_github(readme)
readme = re.sub(r"\n{3,}", "\n\n", readme).rstrip() + "\n"

Path("README.md").write_text(readme, encoding="utf-8")
