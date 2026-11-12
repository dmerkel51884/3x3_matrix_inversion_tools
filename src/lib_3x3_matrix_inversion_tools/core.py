"""Core 3x3 matrix inversion routine.

We use the adjugate/determinant formula rather than Gaussian elimination
because it is closed-form, branch-free, and easy to audit. The trade-off is
numerical stability: for matrices that are close to singular the determinant
is tiny and the inverse entries can blow up. We therefore expose a
``tolerance`` parameter so callers can decide what "near singular" means for
their data.
"""

from __future__ import annotations

from typing import Sequence, Tuple

Matrix3 = Tuple[Tuple[float, float, float], ...]


class SingularMatrixError(ValueError):
    """Raised when a matrix is singular or too close to singular to invert."""


# Default tolerance chosen empirically: it is large enough to catch matrices
# whose determinant is dominated by floating-point noise, yet small enough to
# accept well-conditioned matrices with modestly small determinants.
DEFAULT_TOLERANCE = 1e-12


def _cofactor(m: Sequence[Sequence[float]], row: int, col: int) -> float:
    """Return the cofactor of m at (row, col).

    The cofactor is the signed determinant of the 2x2 submatrix obtained by
    deleting ``row`` and ``col``. We compute it directly rather than via a
    generic recursive minor function because 3x3 is small and the explicit
    arithmetic is faster and easier to read.
    """
    # Build the four entries of the 2x2 minor in row-major order.
    r0, r1 = [i for i in range(3) if i != row]
    c0, c1 = [j for j in range(3) if j != col]
    minor = m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    # Sign is positive when row+col is even, negative when odd.
    return minor if (row + col) % 2 == 0 else -minor


def _determinant(m: Sequence[Sequence[float]]) -> float:
    """Compute the determinant of a 3x3 matrix by cofactor expansion."""
    return (
        m[0][0] * _cofactor(m, 0, 0)
        + m[0][1] * _cofactor(m, 0, 1)
        + m[0][2] * _cofactor(m, 0, 2)
    )


def invert3x3(
    matrix: Sequence[Sequence[float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Matrix3:
    """Return the inverse of a 3x3 matrix.

    Parameters
    ----------
    matrix:
        A sequence of three sequences, each of length three. Values are
        coerced to ``float``.
    tolerance:
        Absolute threshold on ``|det|``. If the determinant is smaller than
        this, :class:`SingularMatrixError` is raised. The default is
        ``1e-12``.

    Returns
    -------
    A tuple of three tuples of three floats.

    Raises
    ------
    SingularMatrixError
        If the matrix is not 3x3 or its determinant is below ``tolerance`` in
        absolute value.
    """
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise SingularMatrixError("matrix must be 3x3")

    # Coerce to float once; this also copies the input so we do not alias it.
    m = tuple(tuple(float(x) for x in row) for row in matrix)

    det = _determinant(m)
    if det == 0.0 or abs(det) < tolerance:
        raise SingularMatrixError(
            "matrix is singular or near-singular (determinant below tolerance)"
        )

    # The inverse is the transpose of the cofactor matrix divided by det.
    # Equivalently, inverse[i][j] = cofactor(m, j, i) / det.
    inv = tuple(
        tuple(_cofactor(m, j, i) / det for j in range(3))
        for i in range(3)
    )
    return inv
