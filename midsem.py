"""
Applied Linear Algebra Lab - Mid-Semester Examination
PAIRSCORE implementation with question-wise answers and unit tests.

Assumptions:
1. X and Z are non-empty, two-dimensional NumPy arrays.
2. The fixed dimensions from the question are dx=6, dz=10, and dh=4.
3. The number of columns in X (lx) and Z (lz) may differ.
4. All calculations use float64.
5. Floating-point comparisons use NumPy testing utilities with tolerances.
6. The algorithm is:
       P = A @ X + a[:, None]
       R = B @ Z + b[:, None]
       S = (R.T @ P) / sqrt(dh)
"""

import unittest
import numpy as np


DX = 6
DZ = 10
DH = 4


def pair_score(X, Z, W):
    """
    Compute the PAIRSCORE matrix.

    Args:
        X (numpy.ndarray): Matrix with shape (6, lx), where lx >= 1.
        Z (numpy.ndarray): Matrix with shape (10, lz), where lz >= 1.
        W (tuple): (A, a, B, b), with shapes (4, 6), (4,), (4, 10), (4,).

    Returns:
        numpy.ndarray: Score matrix S with shape (lz, lx), dtype float64.

    Raises:
        ValueError: If W, array types, or required shapes are invalid.
    """
    if not isinstance(W, tuple) or len(W) != 4:
        raise ValueError("W must be a tuple (A, a, B, b).")

    arrays = (X, Z, *W)
    if any(not isinstance(arr, np.ndarray) for arr in arrays):
        raise ValueError("X, Z, A, a, B, and b must be NumPy arrays.")

    A, a, B, b = W

    if X.ndim != 2 or X.shape[0] != DX or X.shape[1] < 1:
        raise ValueError("X must have shape (6, lx), where lx >= 1.")
    if Z.ndim != 2 or Z.shape[0] != DZ or Z.shape[1] < 1:
        raise ValueError("Z must have shape (10, lz), where lz >= 1.")
    if A.shape != (DH, DX):
        raise ValueError("A must have shape (4, 6).")
    if a.shape != (DH,):
        raise ValueError("a must have shape (4,).")
    if B.shape != (DH, DZ):
        raise ValueError("B must have shape (4, 10).")
    if b.shape != (DH,):
        raise ValueError("b must have shape (4,).")

    # Use float64 for every calculation.
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Question 1(b): P = A X + a 1_lx^T.
    P = A @ X + a[:, None]

    # Question 1(c): R = B Z + b 1_lz^T.
    R = B @ Z + b[:, None]

    # Question 1(d): S = R^T P / sqrt(dh).
    S = (R.T @ P) / np.sqrt(np.float64(DH))
    return S


class TestPairScore(unittest.TestCase):
    """Unit tests for Question 2(c) and the implementation."""

    def setUp(self):
        """Create deterministic arrays for repeatable tests."""
        self.X = np.arange(18, dtype=np.float64).reshape(6, 3) / 10.0
        self.Z = np.arange(50, dtype=np.float64).reshape(10, 5) / 10.0
        self.A = np.arange(24, dtype=np.float64).reshape(4, 6) / 10.0
        self.a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        self.B = np.arange(40, dtype=np.float64).reshape(4, 10) / 10.0
        self.b = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        self.W = (self.A, self.a, self.B, self.b)

    def test_q2a_output_shape_and_dtype(self):
        """Question 2(a): output has shape (lz, lx) and float64 dtype."""
        S = pair_score(self.X, self.Z, self.W)
        self.assertEqual(S.shape, (5, 3))
        self.assertEqual(S.dtype, np.float64)

    def test_q1b_q1c_q1d_direct_formula(self):
        """Questions 1(b)-(d): result matches the mathematical formula."""
        P = self.A @ self.X + self.a[:, None]
        R = self.B @ self.Z + self.b[:, None]
        expected = (R.T @ P) / np.sqrt(np.float64(DH))
        actual = pair_score(self.X, self.Z, self.W)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_q1a_invalid_shapes(self):
        """Question 1(a): invalid shapes must raise ValueError."""
        with self.assertRaises(ValueError):
            pair_score(np.zeros((5, 3)), self.Z, self.W)
        with self.assertRaises(ValueError):
            pair_score(self.X, np.zeros((10, 0)), self.W)
        with self.assertRaises(ValueError):
            pair_score(self.X, self.Z, (np.zeros((6, 4)), self.a, self.B, self.b))
        with self.assertRaises(ValueError):
            pair_score(self.X, self.Z, (self.A, np.zeros((4, 1)), self.B, self.b))
        with self.assertRaises(ValueError):
            pair_score(self.X, self.Z, (self.A, self.a, self.B, np.zeros((3,))))

    def test_q2a_unequal_column_counts(self):
        """Question 2(a): lx and lz may be different positive integers."""
        X = np.zeros((6, 2), dtype=np.float64)
        Z = np.zeros((10, 7), dtype=np.float64)
        S = pair_score(X, Z, self.W)
        self.assertEqual(S.shape, (7, 2))

    def test_q2b_bias_perturbation_structure(self):
        """Question 2(b): changing b to b+delta creates identical rows in change."""
        delta = np.array([0.2, -0.1, 0.3, 0.4], dtype=np.float64)

        original = pair_score(self.X, self.Z, self.W)
        changed_W = (self.A, self.a, self.B, self.b + delta)
        changed = pair_score(self.X, self.Z, changed_W)
        actual_change = changed - original

        P = self.A @ self.X + self.a[:, None]
        expected_row = (delta @ P) / np.sqrt(np.float64(DH))
        expected_change = np.tile(expected_row, (self.Z.shape[1], 1))

        np.testing.assert_allclose(
            actual_change, expected_change, rtol=1e-10, atol=1e-12
        )
        for i in range(1, actual_change.shape[0]):
            np.testing.assert_allclose(
                actual_change[i], actual_change[0], rtol=1e-10, atol=1e-12
            )


def print_question_answers():
    """Print each exam sub-question followed by its concise answer."""
    print("=" * 72)
    print("APPLIED LINEAR ALGEBRA LAB - QUESTION-WISE ANSWERS")
    print("=" * 72)

    print("\\nQUESTION 1(a) [3 marks]")
    print("Question: Validate input and parameter shapes; raise ValueError otherwise.")
    print("Answer: X=(6,lx), Z=(10,lz), A=(4,6), a=(4,), B=(4,10), b=(4,).")
    print("         lx and lz must be positive but may differ. pair_score() checks")
    print("         these conditions and raises ValueError for invalid shapes.")

    print("\\nQUESTION 1(b) [4 marks]")
    print("Question: Compute P = A X + a 1_lx^T.")
    print("Answer:   P = A @ X + a[:, None]. Its shape is (4,lx).")
    print("         a[:, None] reshapes a from (4,) to (4,1), so NumPy broadcasts")
    print("         the bias across every column of A @ X.")

    print("\\nQUESTION 1(c) [4 marks]")
    print("Question: Compute R = B Z + b 1_lz^T.")
    print("Answer:   R = B @ Z + b[:, None]. Its shape is (4,lz).")
    print("         b[:, None] reshapes b to (4,1) and broadcasts across columns.")

    print("\\nQUESTION 1(d) [4 marks]")
    print("Question: Return S = R^T P / sqrt(dh).")
    print("Answer:   S = (R.T @ P) / np.sqrt(4).")
    print("         R.T has shape (lz,4), P has shape (4,lx), so S has shape")
    print("         (lz,lx).")

    print("\\nQUESTION 2(a) [5 marks]")
    print("Question: Analyze the output shape and mathematical operations.")
    print("Answer:   P is (4,lx), R is (4,lz), and R.T @ P is (lz,lx).")
    print("         Each entry S[j,i] is the dot product of column j of R and")
    print("         column i of P, divided by sqrt(dh)=2.")

    print("\\nQUESTION 2(b) [5 marks]")
    print("Question: Replace b by b+delta. Express the change in S and describe it.")
    print("Answer:   R changes to R + delta 1_lz^T.")
    print("         Delta_S = (1_lz (delta^T P)) / sqrt(dh).")
    print("         Delta_S has shape (lz,lx), and every row is identical because")
    print("         the same delta^T P row is repeated for all lz rows.")

    print("\\nQUESTION 2(c) [5 marks]")
    print("Question: Write unit tests verifying parts (a) and (b).")
    print("Answer:   The TestPairScore class tests output shape and dtype, direct")
    print("         formula correctness, unequal lx/lz, invalid shapes, and checks")
    print("         that changing b by delta produces the derived identical-row")
    print("         change matrix. Run the file to execute these tests.")

    print("\\n" + "=" * 72)
    print("NUMERICAL EXAMPLE")
    print("=" * 72)


def main():
    """Print question-wise answers, show an example, and run unit tests."""
    print_question_answers()

    X = np.arange(18, dtype=np.float64).reshape(6, 3) / 10.0
    Z = np.arange(50, dtype=np.float64).reshape(10, 5) / 10.0
    A = np.arange(24, dtype=np.float64).reshape(4, 6) / 10.0
    a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    B = np.arange(40, dtype=np.float64).reshape(4, 10) / 10.0
    b = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)

    S = pair_score(X, Z, (A, a, B, b))
    print("Example S matrix:")
    print(S)
    print("S.shape =", S.shape)

    print("\\nRUNNING UNIT TESTS")
    unittest.main(argv=["ignored"], exit=False, verbosity=2)


if __name__ == "__main__":
    main()
