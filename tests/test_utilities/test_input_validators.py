from __future__ import annotations

import unittest

import numpy as np

from popgames.utilities.input_validators import (
    check_array_dimensions,
    check_array_in_simplex,
    check_array_shape,
    check_callable,
    check_function_signature,
    check_list_length,
    check_scalar_value_bounds,
    check_type,
    check_valid_list,
)


class TestCheckType(unittest.TestCase):
    def test_passes_for_expected_type(self) -> None:
        check_type(1, int, "x")
        check_type("a", (str, bytes), "s")
        check_type(np.array([1, 2]), np.ndarray, "arr")

    def test_raises_typeerror_for_wrong_type(self) -> None:
        with self.assertRaises(TypeError):
            check_type("1", int, "x")

    def test_error_message_contains_arg_name(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            check_type("1", int, "my_arg")
        self.assertIn("my_arg", str(ctx.exception))


class TestCheckScalarValueBounds(unittest.TestCase):
    def test_passes_default_bounds(self) -> None:
        check_scalar_value_bounds(0.0, "x")
        check_scalar_value_bounds(1, "x")
        check_scalar_value_bounds(-5, "x")

    def test_raises_typeerror_if_not_number(self) -> None:
        with self.assertRaises(TypeError):
            check_scalar_value_bounds("1", "x")

    def test_min_value_violation(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            check_scalar_value_bounds(-1, "x", min_value=0)
        self.assertIn("greater or equal", str(ctx.exception))

    def test_max_value_violation(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            check_scalar_value_bounds(2, "x", max_value=1)
        self.assertIn("less or equal", str(ctx.exception))

    def test_strictly_positive_allows_positive(self) -> None:
        check_scalar_value_bounds(0.1, "x", strictly_positive=True)

    def test_strictly_positive_rejects_zero_and_negative(self) -> None:
        with self.assertRaises(ValueError):
            check_scalar_value_bounds(0.0, "x", strictly_positive=True)
        with self.assertRaises(ValueError):
            check_scalar_value_bounds(-0.1, "x", strictly_positive=True)


class TestCheckCallable(unittest.TestCase):
    def test_passes_for_callable(self) -> None:
        check_callable(lambda x: x, "f")

    def test_raises_typeerror_for_non_callable(self) -> None:
        with self.assertRaises(TypeError):
            check_callable(123, "f")

    def test_error_message_contains_arg_name(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            check_callable(123, "my_callable")
        self.assertIn("my_callable", str(ctx.exception))


class TestCheckListLength(unittest.TestCase):
    def test_passes_for_expected_length(self) -> None:
        check_list_length([1, 2, 3], 3, "lst")
        check_list_length("abc", 3, "s")  # any object with __len__ is accepted

    def test_raises_typeerror_if_no_len(self) -> None:
        with self.assertRaises(TypeError):
            check_list_length(123, 1, "x")

    def test_raises_valueerror_if_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            check_list_length([1, 2], 3, "lst")

    def test_error_message_contains_arg_name(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            check_list_length([1, 2], 3, "my_list")
        self.assertIn("my_list", str(ctx.exception))


class TestCheckArrayDimensions(unittest.TestCase):
    def test_passes_for_expected_dim(self) -> None:
        check_array_dimensions(np.zeros((3,)), 1, "x")
        check_array_dimensions(np.zeros((2, 3)), 2, "x")

    def test_raises_typeerror_if_not_array(self) -> None:
        with self.assertRaises(TypeError):
            check_array_dimensions([1, 2, 3], 1, "x")

    def test_raises_valueerror_if_wrong_dim(self) -> None:
        with self.assertRaises(ValueError):
            check_array_dimensions(np.zeros((2, 3)), 1, "x")


class TestCheckArrayShape(unittest.TestCase):
    def test_passes_for_expected_shape(self) -> None:
        check_array_shape(np.zeros((2, 3)), (2, 3), "x")

    def test_raises_valueerror_for_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            check_array_shape(np.zeros((2, 3)), (3, 2), "x")

    def test_raises_if_not_array(self) -> None:
        with self.assertRaises(TypeError):
            check_array_shape([[1, 2]], (1, 2), "x")


class TestCheckArrayInSimplex(unittest.TestCase):
    def test_passes_for_valid_simplex(self) -> None:
        x = np.array([0.2, 0.3, 0.5])
        check_array_in_simplex(x, n=3, m=1.0, arg_name="x")

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            check_array_in_simplex(np.zeros((2, 2)), n=4, m=1.0, arg_name="x")

    def test_rejects_negative_entry(self) -> None:
        x = np.array([-0.1, 0.6, 0.5])
        with self.assertRaises(ValueError):
            check_array_in_simplex(x, n=3, m=1.0, arg_name="x")

    def test_rejects_entry_greater_than_m(self) -> None:
        x = np.array([0.2, 1.2, -0.4])  # has >m and negative; should fail
        with self.assertRaises(ValueError):
            check_array_in_simplex(x, n=3, m=1.0, arg_name="x")

    def test_rejects_sum_not_equal_to_m(self) -> None:
        x = np.array([0.2, 0.2, 0.2])  # sum = 0.6
        with self.assertRaises(ValueError):
            check_array_in_simplex(x, n=3, m=1.0, arg_name="x")

    def test_tolerance_allows_small_sum_error(self) -> None:
        # sum differs by 5e-7, within tolerance 1e-6 => should pass
        x = np.array([0.5, 0.5 + 5e-7])
        check_array_in_simplex(x, n=2, m=1.0, arg_name="x", tolerance=1e-6)

    def test_tolerance_rejects_large_sum_error(self) -> None:
        # sum differs by 2e-6, outside tolerance 1e-6 => should fail
        x = np.array([0.5, 0.5 + 2e-6])
        with self.assertRaises(ValueError):
            check_array_in_simplex(x, n=2, m=1.0, arg_name="x", tolerance=1e-6)


class TestCheckValidList(unittest.TestCase):
    def test_passes_for_valid_list(self) -> None:
        check_valid_list([1, 2, 3], length=3, internal_type=int, name="x")
        check_valid_list(["a", "b"], length=2, internal_type=str, name="x")

    def test_rejects_non_list(self) -> None:
        with self.assertRaises(TypeError):
            check_valid_list((1, 2), length=2, internal_type=int, name="x")

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            check_valid_list([1, 2], length=3, internal_type=int, name="x")

    def test_rejects_wrong_internal_type(self) -> None:
        with self.assertRaises(ValueError):
            check_valid_list([1, "2", 3], length=3, internal_type=int, name="x")

    def test_strictly_positive_passes(self) -> None:
        check_valid_list(
            [1, 2, 3], length=3, internal_type=int, name="x", strictly_positive=True
        )

    def test_strictly_positive_rejects_zero_or_negative(self) -> None:
        with self.assertRaises(ValueError):
            check_valid_list(
                [1, 0, 3], length=3, internal_type=int, name="x", strictly_positive=True
            )
        with self.assertRaises(ValueError):
            check_valid_list(
                [1, -1, 3],
                length=3,
                internal_type=int,
                name="x",
                strictly_positive=True,
            )

    def test_strictly_positive_rejects_non_numeric_even_if_internal_type_matches(
        self,
    ) -> None:
        # internal_type is object, but strictly_positive requires numbers.Number and > 0
        with self.assertRaises(ValueError):
            check_valid_list(
                ["a", "b"],
                length=2,
                internal_type=object,
                name="x",
                strictly_positive=True,
            )


class TestCheckFunctionSignature(unittest.TestCase):
    def test_passes_for_valid_function(self) -> None:
        def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a + b

        check_function_signature(
            f,
            expected_input_shapes=[(2, 3), (2, 3)],
            expected_output_shape=(2, 3),
            name="f",
        )

    def test_rejects_non_callable(self) -> None:
        with self.assertRaises(TypeError):
            check_function_signature(
                123,
                expected_input_shapes=[(2,)],
                expected_output_shape=(2,),
                name="f",
            )

    def test_rejects_function_that_raises(self) -> None:
        def f_bad(a: np.ndarray) -> np.ndarray:
            raise RuntimeError("boom")

        with self.assertRaises(ValueError) as ctx:
            check_function_signature(
                f_bad,
                expected_input_shapes=[(2,)],
                expected_output_shape=(2,),
                name="f_bad",
            )
        self.assertIn("raised an error", str(ctx.exception))

    def test_rejects_wrong_output_type(self) -> None:
        def f_bad(a: np.ndarray) -> list[float]:
            return [1.0, 2.0]  # not an ndarray

        with self.assertRaises(ValueError) as ctx:
            check_function_signature(
                f_bad,
                expected_input_shapes=[(2,)],
                expected_output_shape=(2,),
                name="f_bad",
            )
        self.assertIn("must return as output a np.ndarray", str(ctx.exception))

    def test_rejects_wrong_output_shape(self) -> None:
        def f_bad(a: np.ndarray) -> np.ndarray:
            return np.ones((3,))  # wrong shape

        with self.assertRaises(ValueError) as ctx:
            check_function_signature(
                f_bad,
                expected_input_shapes=[(2,)],
                expected_output_shape=(2,),
                name="f_bad",
            )
        self.assertIn("shape", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
