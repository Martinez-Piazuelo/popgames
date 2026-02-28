from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from popgames.population_game import PopulationGame, SinglePopulationGame


def _fitness_ok(x: np.ndarray) -> np.ndarray:
    # expects (n,1) -> returns (n,1)
    return 2.0 * x


def _fitness_wrong_shape(x: np.ndarray) -> np.ndarray:
    # wrong output shape
    return np.zeros((x.shape[0],))


def _fitness_raises(x: np.ndarray) -> np.ndarray:
    raise RuntimeError("boom")


class TestPopulationGameInit(unittest.TestCase):
    def test_minimal_valid_init_sets_fields_and_defaults(self) -> None:
        g = PopulationGame(
            num_populations=2,
            num_strategies=[2, 3],
            fitness_function=_fitness_ok,
        )
        self.assertEqual(g.num_populations, 2)
        self.assertEqual(g.num_strategies, [2, 3])
        self.assertEqual(g.n, 5)

        # default masses
        self.assertEqual(g.masses, [1.0, 1.0])

        # no constraints
        self.assertIsNone(g.A_eq)
        self.assertIsNone(g.b_eq)
        self.assertIsNone(g.A_ineq)
        self.assertIsNone(g.b_ineq)

        self.assertEqual(g.d_eq, 0)
        self.assertEqual(g.d_ineq, 0)

    def test_rejects_non_int_num_populations(self) -> None:
        with self.assertRaises(TypeError):
            PopulationGame(
                num_populations=2.0,  # type: ignore[arg-type]
                num_strategies=[2, 3],
                fitness_function=_fitness_ok,
            )

    def test_rejects_non_positive_num_populations(self) -> None:
        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=0,
                num_strategies=[],
                fitness_function=_fitness_ok,
            )

    def test_rejects_invalid_num_strategies_list(self) -> None:
        # wrong length
        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=2,
                num_strategies=[2],
                fitness_function=_fitness_ok,
            )
        # non-positive
        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=2,
                num_strategies=[2, 0],
                fitness_function=_fitness_ok,
            )
        # wrong internal type
        with self.assertRaises(ValueError):
            PopulationGame(  # type: ignore[list-item]
                num_populations=2,
                num_strategies=[2, "3"],
                fitness_function=_fitness_ok,
            )

    def test_masses_optional_and_validated(self) -> None:
        g = PopulationGame(
            num_populations=2,
            num_strategies=[1, 2],
            fitness_function=_fitness_ok,
            masses=[2.0, 3.5],
        )
        self.assertEqual(g.masses, [2.0, 3.5])

        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=2,
                num_strategies=[1, 2],
                fitness_function=_fitness_ok,
                masses=[1.0],  # wrong length
            )

        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=2,
                num_strategies=[1, 2],
                fitness_function=_fitness_ok,
                masses=[1.0, -1.0],  # not strictly positive
            )

    def test_fitness_function_signature_is_validated(self) -> None:
        # wrong output shape
        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=1,
                num_strategies=[3],
                fitness_function=_fitness_wrong_shape,
            )

        # function raises when called with dummy input
        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=1,
                num_strategies=[3],
                fitness_function=_fitness_raises,
            )

    def test_constraints_are_reshaped_correctly(self) -> None:
        # n = 3
        A_eq = np.array([[1, 0, 0], [0, 1, 0]])  # must be ndarray
        b_eq = np.ndarray([1, 2])  # allowed: (numbers.Number, np.ndarray)
        A_ineq = np.array([[1, 1, 1]])  # must be ndarray
        b_ineq = 5  # allowed: (numbers.Number, np.ndarray)

        g = PopulationGame(
            num_populations=1,
            num_strategies=[3],
            fitness_function=_fitness_ok,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
        )

        self.assertIsInstance(g.A_eq, np.ndarray)
        self.assertIsInstance(g.b_eq, np.ndarray)
        self.assertIsInstance(g.A_ineq, np.ndarray)
        self.assertIsInstance(g.b_ineq, np.ndarray)

        self.assertEqual(g.A_eq.shape, (2, 3))
        self.assertEqual(g.b_eq.shape, (2, 1))
        self.assertEqual(g.A_ineq.shape, (1, 3))
        self.assertEqual(g.b_ineq.shape, (1, 1))

        self.assertEqual(g.d_eq, 2)
        self.assertEqual(g.d_ineq, 1)

    def test_constraints_reject_non_ndarray_A_matrices(self) -> None:
        with self.assertRaises(TypeError):
            PopulationGame(
                num_populations=1,
                num_strategies=[3],
                fitness_function=_fitness_ok,
                A_eq=[[1, 0, 0], [0, 1, 0]],  # type: ignore[assignment]
                b_eq=np.ones((2, 1)),
            )

    def test_constraints_reject_bad_reshape(self) -> None:
        # n=3, but A_eq has 4 cols -> cannot reshape to (-1, 3)
        A_eq_bad = np.ones((2, 4))
        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=1,
                num_strategies=[3],
                fitness_function=_fitness_ok,
                A_eq=A_eq_bad,
                b_eq=np.ones((2, 1)),
            )

    def test_constraints_reject_wrong_b_shapes_when_rows_present(self) -> None:
        # Provide A_eq with 2 rows but b_eq with wrong shape (3,1)
        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=1,
                num_strategies=[3],
                fitness_function=_fitness_ok,
                A_eq=np.ones((2, 3)),
                b_eq=np.ones((3, 1)),
            )

        # Provide A_ineq with 2 rows but b_ineq with wrong shape (1,1)
        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=1,
                num_strategies=[3],
                fitness_function=_fitness_ok,
                A_ineq=np.ones((2, 3)),
                b_ineq=np.ones((1, 1)),
            )

    def test_fitness_lipschitz_constant_optional_and_validated(self) -> None:
        g = PopulationGame(
            num_populations=1,
            num_strategies=[2],
            fitness_function=_fitness_ok,
            fitness_lipschitz_constant=1.23,
        )
        self.assertEqual(g._fitness_lipschitz_constant, 1.23)

        with self.assertRaises(ValueError):
            PopulationGame(
                num_populations=1,
                num_strategies=[2],
                fitness_function=_fitness_ok,
                fitness_lipschitz_constant=0.0,
            )


class TestPopulationGameMethods(unittest.TestCase):
    @patch("popgames.population_game.fbos")
    def test_compute_gne_validates_inputs_and_calls_fbos(self, fbos_mock) -> None:
        g = PopulationGame(
            num_populations=1,
            num_strategies=[2],
            fitness_function=_fitness_ok,
        )

        fbos_mock.return_value = np.array([0.5, 0.5]).reshape(2, 1)

        out = g.compute_gne(max_iter=123, tolerance=1e-4)

        fbos_mock.assert_called_once()
        # called with keyword args
        kwargs = fbos_mock.call_args.kwargs
        self.assertIs(kwargs["population_game"], g)
        self.assertEqual(kwargs["max_iter"], 123)
        self.assertEqual(kwargs["tolerance"], 1e-4)

        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (2, 1))

        with self.assertRaises(TypeError):
            g.compute_gne(max_iter=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            g.compute_gne(max_iter=0)
        with self.assertRaises(ValueError):
            g.compute_gne(tolerance=0.0)

    @patch("popgames.population_game.compute_vertices")
    def test_compute_polyhedron_vertices_delegates(self, compute_vertices_mock) -> None:
        g = PopulationGame(
            num_populations=1,
            num_strategies=[2],
            fitness_function=_fitness_ok,
        )
        verts = np.array([[0.0, 1.0], [1.0, 0.0]])
        compute_vertices_mock.return_value = verts

        out = g.compute_polyhedron_vertices()

        compute_vertices_mock.assert_called_once_with(g)
        np.testing.assert_allclose(out, verts)


class TestSinglePopulationGame(unittest.TestCase):
    def test_single_population_game_basic(self) -> None:
        g = SinglePopulationGame(
            num_strategies=3,
            fitness_function=_fitness_ok,
        )
        self.assertIsInstance(g, PopulationGame)
        self.assertEqual(g.num_populations, 1)
        self.assertEqual(g.num_strategies, [3])
        self.assertEqual(g.n, 3)
        self.assertEqual(g.masses, [1.0])

    def test_single_population_game_with_mass(self) -> None:
        g = SinglePopulationGame(
            num_strategies=2,
            fitness_function=_fitness_ok,
            mass=4.2,
        )
        self.assertEqual(g.masses, [4.2])

    def test_single_population_constraints_pass_through(self) -> None:
        A_eq = np.array([[1, 1]])
        b_eq = np.array([[1]])
        g = SinglePopulationGame(
            num_strategies=2,
            fitness_function=_fitness_ok,
            A_eq=A_eq,
            b_eq=b_eq,
        )
        self.assertEqual(g.A_eq.shape, (1, 2))
        self.assertEqual(g.b_eq.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
