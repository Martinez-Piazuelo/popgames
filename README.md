# PopGames

A Python package to model and simulate population games.

[![PyPI version](https://img.shields.io/pypi/v/popgames.svg)](https://pypi.org/project/popgames/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/popgames.svg)](https://pypi.org/project/popgames/)
[![docs](https://img.shields.io/badge/docs-online-success)](https://martinez-piazuelo.github.io/popgames/)

---

## Installation

Install the latest version from PyPI:

```bash
pip install popgames
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/Martinez-Piazuelo/popgames.git
```

## Documentation

Full API docs and usage examples are available at:

[https://martinez-piazuelo.github.io/popgames/](https://martinez-piazuelo.github.io/popgames/)

---

## Contributing

Contributions are welcome. If you would like to improve `popgames`, please follow the workflow below.

### 1. Fork and clone the repository

Fork the repository on GitHub, then clone your fork locally:

```bash
git clone https://github.com/<your-username>/popgames.git
cd popgames
```

Create a new branch for your changes:

```bash
git checkout -b your-feature-branch-name
```

---

### 2. Set up the development environment

This project uses `uv` for dependency and environment management. 

Install `uv` if it is not already installed:

```bash
pip install uv
```

Install the development environment with:

```bash
uv sync
```

This installs the package together with the default dependency groups used for development and documentation.

The repository also uses a **Taskfile** to define common development commands. The Task runner (`go-task`) is 
installed automatically as part of the development dependencies.

---

### 3. Run the test suite

Before submitting a pull request, ensure all tests pass:

```bash
uv run task test
```

---

### 4. Format and check the code

Please ensure your code is properly formatted and passes all checks before opening a pull request:

```bash
uv run task format
uv run task check
```

---

### 5. Submit a pull request

Push your branch to your fork and open a Pull Request against the main repository.

Please include:

* A clear description of the changes
* Any relevant issue references
* Tests for new functionality when applicable
* Documentation updates if needed

---

## Development notes

* Keep contributions focused and minimal.
* Follow the existing project structure and coding style.
* Run formatting, linting, and tests locally before submitting your pull request.
