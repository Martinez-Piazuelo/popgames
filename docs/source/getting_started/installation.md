## Installation

PopGames is available on PyPI and can be installed using `pip`.

### Install from PyPI

```bash
pip install popgames
```

This installs the latest released version together with its dependencies.

### Install the development version

If you want the latest features, you can install the development version directly from GitHub:

```bash
pip install git+https://github.com/Martinez-Piazuelo/popgames.git
```

### Optional: Numba backend

To speed up large simulations with the optional [Numba](https://numba.pydata.org/) backend, install the
``numba`` extra:

```bash
pip install "popgames[numba]"
```

Numba is only needed when a simulator is created with ``backend="numba"``.

### Verify the installation

You can verify that PopGames is installed correctly by running:

```python
import popgames
print(popgames.__version__)
```

If no errors occur, the installation was successful.