import time
import math
import numpy as np
from vec import Vec

def run_tests():
    """Unit tests to ensure fundamental Vec methods work correctly."""
    print("--- Running Fundamental Unit Tests ---")
    v1 = Vec([1, 2, 3])
    v2 = Vec([4, 5, 6])
    
    assert (v1 + v2).elements == [5, 7, 9], "Addition failed"
    assert (v2 - v1).elements == [3, 3, 3], "Subtraction failed"
    assert (-v1).elements == [-1, -2, -3], "Negation failed"
    assert (2 * v1).elements == [2, 4, 6], "Right multiplication failed"
    assert (1 + v1).elements == [2, 3, 4], "Right addition failed"
    
    v3 = Vec([1, 1, 1])
    v3 += v1
    assert v3.elements == [2, 3, 4], "In-place addition failed"
    v3 *= 2
    assert v3.elements == [4, 6, 8], "In-place multiplication failed"
    
    assert len(Vec.zeros(10)) == 10, "Zeros generator failed"
    assert sum(Vec.ones(5).elements) == 5, "Ones generator failed"
    assert Vec([3, 4]).norm() == 5.0, "Norm calculation failed"
    
    print("Fundamental unit tests passed successfully!\n")

def test_statistics():
    """Unit tests verifying the properties of mean, demean, and std."""
    print("--- Running Statistical Property Tests ---")
    
    # Using the vector from the manual worksheet: x = (1, 3, 5, 7)
    x = Vec([1, 3, 5, 7])
    
    # 1. Test Mean
    assert x.mean() == 4.0, f"Expected mean 4.0, got {x.mean()}"
    
    # 2. Test Demean
    x_tilde = x.demean()
    assert x_tilde.elements == [-3.0, -1.0, 1.0, 3.0], f"Demean failed, got {x_tilde.elements}"
    
    # Property A: The mean of a demeaned vector is always 0
    assert x_tilde.mean() == 0.0, "Mean of demeaned vector must be 0"
    
    # 3. Test Standard Deviation
    # Variance of (-3, -1, 1, 3) is (9 + 1 + 1 + 9) / 4 = 20 / 4 = 5
    # Standard deviation is sqrt(5) ≈ 2.23607
    expected_std = round(math.sqrt(5), 5)
    assert x.std() == expected_std, f"Expected std {expected_std}, got {x.std()}"
    
    # Property B: The standard deviation of a demeaned vector is equal to 
    # the standard deviation of the original vector.
    assert x_tilde.std() == x.std(), "Standard deviation should not change after demeaning"
    
    print("Statistical property tests passed successfully!\n")

def benchmark_performance():
    """Measures and compares Vec vs NumPy performance on large vectors."""
    print("--- Running Performance Benchmarks ---")
    dimensions = [2000, 8000, 32000, 64000]
    
    print(f"{'Size':<10} | {'Custom Vec Time (s)':<25} | {'NumPy Time (s)':<20}")
    print("-" * 60)
    
    for n in dimensions:
        v_a = Vec.uniform(n)
        v_b = Vec.uniform(n)
        
        start_time = time.perf_counter()
        v_c = v_a + v_b
        _ = v_c.norm()
        _ = 5.5 * v_c
        vec_time = time.perf_counter() - start_time
        
        np_a = np.random.uniform(0, 1, n)
        np_b = np.random.uniform(0, 1, n)
        
        start_time = time.perf_counter()
        np_c = np_a + np_b
        _ = np.linalg.norm(np_c)
        _ = 5.5 * np_c
        np_time = time.perf_counter() - start_time
        
        print(f"{n:<10} | {vec_time:<25.6f} | {np_time:<20.6f}")

if __name__ == "__main__":
    run_tests()
    test_statistics()
    benchmark_performance()
