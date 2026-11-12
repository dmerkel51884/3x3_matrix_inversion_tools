# 3x3_matrix_inversion_tools

Compute the inverse of a 3x3 matrix using the adjugate and determinant.

```python
from lib_3x3_matrix_inversion_tools import invert3x3, SingularMatrixError

m = ((1.0, 2.0, 3.0), (0.0, 1.0, 4.0), (5.0, 6.0, 0.0))
inv = invert3x3(m)
print(inv)
# ((-24.0, 18.0, 5.0), (20.0, -15.0, -4.0), (-5.0, 4.0, 1.0))

try:
    invert3x3(((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (7.0, 8.0, 9.0)))
except SingularMatrixError:
    print("that matrix has no inverse")
```

## Why this exists

Most projects that need a 3x3 inverse either pull in a full linear algebra
package or hand-roll a Gaussian elimination loop that is harder to audit than
the closed-form formula. This library is the middle ground: one function, no
dependencies, and a single explicit tolerance parameter so the caller decides
what counts as singular. The trade-off is that the adjugate formula is less
numerically stable than pivoted elimination for ill-conditioned matrices; if
your matrices are near-singular, pass a larger ``tolerance`` and expect
accordingly larger errors.

## The awkward edge

The determinant of a near-singular matrix is a small number computed from
larger numbers, so its sign and magnitude can be dominated by floating-point
rounding. The default tolerance is ``1e-12``. If your data routinely produces
determinants around ``1e-10``, the default will accept them and may return a
noisy inverse; raise the tolerance to match your precision budget.

## API

- ``invert3x3(matrix, tolerance=1e-12)`` returns a 3x3 tuple of tuples of
  floats. ``matrix`` may be any sequence of three sequences of three numbers.
  Raises ``SingularMatrixError`` if the shape is wrong or ``abs(det) <
  tolerance``.
- ``SingularMatrixError`` is a subclass of ``ValueError`` raised for both
  shape errors and singular matrices.
