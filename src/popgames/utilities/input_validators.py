from __future__ import annotations

import numbers
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from typing import Any, Callable


def check_type(arg: Any, expected_type: type | tuple[type, ...], arg_name: str) -> None:
    """
    Check that ``arg`` is an instance of ``expected_type``.

    Args:
        arg: The value to validate.
        expected_type: A type or tuple of types that ``arg`` must be an instance of.
        arg_name: Name of the argument (used in error messages).
    """
    if not isinstance(arg, expected_type):
        raise TypeError(
            f"Expected {arg_name} to be of type {expected_type}, but got {type(arg).__name__} instead."
        )


def check_scalar_value_bounds(
    arg: Any,
    arg_name: str,
    min_value: float = float("-inf"),
    max_value: float = float("inf"),
    strictly_positive: bool = False,
) -> None:
    """
    Check that ``arg`` is a scalar number within bounds.

    Args:
        arg: The scalar value to validate.
        arg_name: Name of the argument (used in error messages).
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).
        strictly_positive: If True, requires ``arg > 0``.
    """
    check_type(arg, numbers.Number, arg_name)

    # Keep comparisons robust for numeric-like inputs
    x = float(arg)
    if x < min_value:
        raise ValueError(
            f"Expected {arg_name} to be greater or equal than {min_value}, but got {arg_name}={arg} instead."
        )
    if x > max_value:
        raise ValueError(
            f"Expected {arg_name} to be less or equal than {max_value}, but got {arg_name}={arg} instead."
        )
    if strictly_positive and x <= 0.0:
        raise ValueError(
            f"Expected {arg_name} to be strictly positive, but got {arg_name}={arg} instead."
        )


def check_callable(arg: Any, arg_name: str) -> None:
    """
    Check that ``arg`` is callable.

    Args:
        arg: The value to validate.
        arg_name: Name of the argument (used in error messages).
    """
    if not callable(arg):
        raise TypeError(
            f"Expected {arg_name} to be a callable object, but got {type(arg).__name__} instead."
        )


def check_list_length(arg: Any, expected_length: int, arg_name: str) -> None:
    """
    Check that ``arg`` has a specific length.

    Args:
        arg: An object that must support ``len(arg)``.
        expected_length: Expected length.
        arg_name: Name of the argument (used in error messages).
    """
    try:
        n = len(arg)
    except TypeError as e:
        raise TypeError(
            f"Expected '{arg_name}' to have a length, but got {type(arg).__name__} instead."
        ) from e

    if n != expected_length:
        raise ValueError(
            f"Expected '{arg_name}' to be a list of length {expected_length}, but got length {n} instead."
        )


def check_array_dimensions(arg: Any, expected_dim: int, arg_name: str) -> None:
    """
    Check that ``arg`` is a NumPy array with the expected number of dimensions.

    Args:
        arg: The value to validate.
        expected_dim: Expected number of dimensions.
        arg_name: Name of the argument (used in error messages).
    """
    check_type(arg, np.ndarray, arg_name)
    if arg.ndim != expected_dim:
        raise ValueError(
            f"Expected {arg_name} to be a {expected_dim}-dimensional array, but got a {arg.ndim}-dimensional array instead."
        )


def check_array_shape(arg: Any, expected_shape: tuple[int, ...], arg_name: str) -> None:
    """
    Check that ``arg`` is a NumPy array with the expected shape.

    Args:
        arg: The value to validate.
        expected_shape: Expected shape.
        arg_name: Name of the argument (used in error messages).
    """
    check_array_dimensions(arg, len(expected_shape), arg_name)
    if arg.shape != expected_shape:
        raise ValueError(
            f"Expected {arg_name} to be of shape {expected_shape}, but got shape {arg.shape} instead."
        )


def check_array_in_simplex(
    arg: Any,
    n: int,
    m: float,
    arg_name: str,
    tolerance: float = 1e-6,
) -> None:
    """
    Check that ``arg`` is a 1D array of length ``n`` in the simplex defined by:
      - ``0<=arg[i]<=m`` for all i
      - ``sum(arg)==m`` within ``tolerance``

    Args:
        arg: The value to validate.
        n: Expected length of the vector.
        m: Simplex mass / total sum, and also an upper bound for entries.
        arg_name: Name of the argument (used in error messages).
        tolerance: Absolute tolerance for the sum constraint.
    """
    check_array_shape(arg, (n,), arg_name)

    x = arg.astype(float, copy=False)

    if np.any(x < 0.0) or np.any(x > m):
        for i in range(n):
            check_scalar_value_bounds(
                x[i], f"{arg_name}[{i}]", min_value=0, max_value=m
            )

    total = float(np.sum(x))
    if total > m + tolerance or total < m - tolerance:
        raise ValueError(
            f"Expected sum({arg_name})={m}, but got sum({arg_name})={total} instead (tolerance={tolerance})."
        )


def check_valid_list(
    arg: Any,
    length: int,
    internal_type: type | tuple[type, ...],
    name: str,
    strictly_positive: bool = False,
) -> None:
    """
    Check that ``arg`` is a list of a given length, with elements of ``internal_type``,
    optionally requiring all elements to be strictly positive.

    Args:
        arg: The value to validate.
        length: Expected list length.
        internal_type: Expected type (or tuple of types) for each element.
        name: Name of the argument (used in error messages).
        strictly_positive: If True, requires each element to be numeric and > 0.
    """
    check_type(arg, list, name)
    check_list_length(arg, length, name)

    if strictly_positive:
        cond = all(
            isinstance(i, internal_type) and isinstance(i, numbers.Number) and i > 0
            for i in arg
        )
        if not cond:
            raise ValueError(
                f"Input {name} is expected to be a list of {length} positive {internal_type}s. Got {arg} instead."
            )
    else:
        cond = all(isinstance(i, internal_type) for i in arg)
        if not cond:
            raise ValueError(
                f"Input {name} is expected to be a list of {length} {internal_type}s. Got {arg} instead."
            )


def check_function_signature(
    arg: object | Callable,
    expected_input_shapes: list[tuple[int, ...]],
    expected_output_shape: tuple[int, ...],
    name: str,
) -> None:
    """
    Check that a callable can be invoked with dummy NumPy array inputs of given shapes,
    and returns a NumPy array of the expected output shape.

    Args:
        arg: The candidate callable to validate.
        expected_input_shapes: List of shapes for the dummy input arrays.
        expected_output_shape: Required shape of the output array.
        name: Name of the argument (used in error messages).
    """
    check_callable(arg, name)

    inputs: list[np.ndarray] = [np.ones(shape=shape) for shape in expected_input_shapes]

    try:
        out = arg(*inputs)
    except Exception as e:
        raise ValueError(
            f"Provided {name} raised an error when called with an input of {len(expected_input_shapes)} "
            f"np.ndarrays of shapes={expected_input_shapes}"
        ) from e

    try:
        check_type(out, np.ndarray, name + "(dummy_input)")
        check_array_shape(out, expected_output_shape, name + "(dummy_input)")
    except Exception as e:
        raise ValueError(
            f"Provided {name} must return as output a np.ndarray of shape={expected_output_shape} "
            f"when called with an input of {len(expected_input_shapes)} np.ndarrays of shapes={expected_input_shapes}"
        ) from e
