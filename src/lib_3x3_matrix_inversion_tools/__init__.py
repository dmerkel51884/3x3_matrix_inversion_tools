"""3x3 matrix inversion tools.

This package provides a single function, :func:`invert3x3`, that computes the
inverse of a 3x3 matrix using the adjugate and determinant. It is deliberately
small and dependency-free so it can be dropped into any project without
pulling in a numerical stack.
"""

from .core import invert3x3, SingularMatrixError

__all__ = ["invert3x3", "SingularMatrixError"]
