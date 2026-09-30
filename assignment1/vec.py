import sys
import math
import random
from typing import Self

class Vec:
    """A custom vector class implementation for educational purposes."""
    
    def __init__(self, src=None) -> Self:
        """
        Initializes a vector. If no source is provided, creates an empty vector.
        Otherwise, copies elements from the source iterable and validates they are numbers.
        """
        if src is None:
            self.elements = []
        else:
            elements = list(src)
            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")
            self.elements = elements

    def __add__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError("Type error - vectors must be of same dimensions")
        return Vec([round(x + y, 5) for x, y in zip(self.elements, t.elements)])

    def __rmul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        return Vec([round(x * scalar, 5) for x in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        for i, val in enumerate(self.elements):
            self.elements[i] = round(val * scalar, 5)
        return self

    def __repr__(self) -> str:
        return repr(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __sub__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError("Type error - vectors must be of same dimensions")
        return Vec([round(x - y, 5) for x, y in zip(self.elements, t.elements)])

    def __neg__(self) -> Self:
        return Vec([round(-x, 5) for x in self.elements])

    def __radd__(self, other: int | float) -> Self:
        if not isinstance(other, (int, float)):
            raise TypeError(f"Expected number for scalar addition: {type(other)}")
        return Vec([round(other + x, 5) for x in self.elements])

    def __iadd__(self, other: Self) -> Self:
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")
        if len(self.elements) != len(other):
            raise TypeError("Type error - vectors must be of same dimensions")
        for i in range(len(self.elements)):
            self.elements[i] = round(self.elements[i] + other.elements[i], 5)
        return self

    @staticmethod
    def zeros(n: int) -> Self:
        if n <= 0:
            raise ValueError("n must be > 0")
        return Vec([0.0] * n)

    @staticmethod
    def ones(n: int) -> Self:
        if n <= 0:
            raise ValueError("n must be > 0")
        return Vec([1.0] * n)

    @staticmethod
    def uniform(n: int) -> Self:
        if n <= 0:
            raise ValueError("n must be > 0")
        return Vec([round(random.uniform(0, 1), 5) for _ in range(n)])

    def norm(self) -> float:
        return round(math.sqrt(sum(x**2 for x in self.elements)), 5)

    # --- New Statistical Methods ---

    def mean(self) -> float:
        """Returns the mean (average) of the vector entries."""
        if not self.elements:
            raise ValueError("Cannot compute mean of an empty vector")
        return sum(self.elements) / len(self.elements)

    def demean(self) -> Self:
        """Returns a new vector with the mean subtracted from each entry."""
        mu = self.mean()
        return Vec([round(x - mu, 5) for x in self.elements])

    def std(self) -> float:
        """
        Returns the standard deviation of the vector entries using the 
        square root of the average of the squared demeaned entries.
        """
        if not self.elements:
            raise ValueError("Cannot compute std of an empty vector")
        
        # 1. Demean the vector
        demeaned_vec = self.demean()
        
        # 2. Average of the squared deviations (Variance)
        variance = sum(x**2 for x in demeaned_vec.elements) / len(self.elements)
        
        # 3. Square root of variance
        return round(math.sqrt(variance), 5)
if __name__ == "__main__":    
    z1 = Vec.zeros(10)
    v1 = Vec([0, 1, 1.03])
    print(v1)
    v3 = 2.2 * v1
    v3 *= 5
    v3 = 1 + v3
    print(v3)
    v2 = v1 + v3
    print(v1 + v3)
    print(-(v1 + v3))