import math
import unittest

from lib_3x3_matrix_inversion_tools import invert3x3, SingularMatrixError


def assert_matrix_close(testcase, got, want, places=9):
    testcase.assertEqual(len(got), 3)
    for i in range(3):
        testcase.assertEqual(len(got[i]), 3)
        for j in range(3):
            testcase.assertAlmostEqual(got[i][j], want[i][j], places=places)


class TestInvert3x3(unittest.TestCase):
    def test_identity(self):
        ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        assert_matrix_close(self, invert3x3(ident), ident)

    def test_diagonal(self):
        m = ((2.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 0.5))
        want = ((0.5, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 2.0))
        assert_matrix_close(self, invert3x3(m), want)

    def test_known_matrix(self):
        # Classic textbook example with a clean integer inverse.
        m = ((1.0, 2.0, 3.0), (0.0, 1.0, 4.0), (5.0, 6.0, 0.0))
        want = (
            (-24.0, 18.0, 5.0),
            (20.0, -15.0, -4.0),
            (-5.0, 4.0, 1.0),
        )
        assert_matrix_close(self, invert3x3(m), want)

    def test_inverse_times_original_is_identity(self):
        m = ((4.0, 7.0, 2.0), (3.0, 6.0, 1.0), (2.0, 5.0, 3.0))
        inv = invert3x3(m)
        product = tuple(
            tuple(
                sum(m[i][k] * inv[k][j] for k in range(3))
                for j in range(3)
            )
            for i in range(3)
        )
        ident = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        assert_matrix_close(self, product, ident, places=9)

    def test_singular_zero_row(self):
        m = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (4.0, 5.0, 6.0))
        with self.assertRaises(SingularMatrixError):
            invert3x3(m)

    def test_singular_duplicate_rows(self):
        m = ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        with self.assertRaises(SingularMatrixError):
            invert3x3(m)

    def test_near_singular_below_tolerance(self):
        # Determinant is exactly zero in exact arithmetic because row 1 = 2*row 0.
        m = ((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (7.0, 8.0, 9.0))
        with self.assertRaises(SingularMatrixError):
            invert3x3(m)

    def test_near_singular_respects_custom_tolerance(self):
        # Determinant is 1e-10, which is above the default 1e-12 but below 1e-9.
        m = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1e-10))
        # Default tolerance accepts it.
        invert3x3(m)
        # A stricter tolerance rejects it.
        with self.assertRaises(SingularMatrixError):
            invert3x3(m, tolerance=1e-9)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(SingularMatrixError):
            invert3x3(((1.0, 2.0), (3.0, 4.0)))
        with self.assertRaises(SingularMatrixError):
            invert3x3(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))

    def test_accepts_integers(self):
        m = ((1, 0, 0), (0, 2, 0), (0, 0, 4))
        want = ((1.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.25))
        assert_matrix_close(self, invert3x3(m), want)

    def test_does_not_mutate_input(self):
        m = ((1.0, 2.0, 3.0), (0.0, 1.0, 4.0), (5.0, 6.0, 0.0))
        snapshot = tuple(tuple(row) for row in m)
        invert3x3(m)
        self.assertEqual(m, snapshot)

    def test_tolerance_must_be_positive(self):
        # A zero tolerance would let a truly singular matrix through if det is
        # exactly zero, but our check is ``abs(det) < tolerance``, so with
        # tolerance=0 a singular matrix raises only when det == 0 exactly.
        # We document and test that behaviour rather than inventing a new rule.
        m = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(SingularMatrixError):
            invert3x3(m, tolerance=0.0)


if __name__ == "__main__":
    unittest.main()
