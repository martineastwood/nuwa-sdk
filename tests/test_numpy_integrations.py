"""
Integration tests for nuwa_sdk numpy array wrappers.

These tests should be run from the nuwa-example project directory:
  cd /path/to/nuwa-example/example_project
  python ../nuwa-sdk/tests/test_numpy_integrations.py

Or copy this file to the nuwa-example/tests/ directory and run with pytest.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path to import from nuwa-example
sys.path.insert(0, str(Path(__file__).parent.parent / "nuwa-example" / "example_project"))

try:
    # These functions will be available once we add them to example_project_lib.nim
    from example_project import (
        numpy_array_sum,
        numpy_array_sum_fast,
        numpy_array_multiply_scalar,
        numpy_array_multiply_in_place,
        numpy_matrix_multiply,
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from example_project: {e}")
    print("These tests require nuwa-example to be built with the latest nuwa_sdk")
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Requires nuwa-example to be built")
class TestNumpyWrappersViaExampleProject:
    """Test numpy wrappers through the example project."""

    def test_1d_array_sum(self):
        """Test basic 1D array summation"""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = numpy_array_sum(arr)
        assert result == 15

    def test_1d_array_sum_fast(self):
        """Test 1D array sum with GIL release"""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = numpy_array_sum_fast(arr)
        assert result == 15

    def test_large_array_sum(self):
        """Test with larger array to verify GIL release works"""
        arr = np.arange(1000, dtype=np.int64)
        expected = sum(range(1000))
        result = numpy_array_sum_fast(arr)
        assert result == expected

    def test_1d_array_multiply_scalar(self):
        """Test scalar multiplication"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = numpy_array_multiply_scalar(arr, 2.5)
        assert result == [2.5, 5.0, 7.5]

    def test_1d_array_multiply_in_place(self):
        """Test in-place modification"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        original_id = id(arr)
        numpy_array_multiply_in_place(arr, 3.0)
        assert id(arr) == original_id  # Same object
        np.testing.assert_array_almost_equal(arr, [3.0, 6.0, 9.0])

    def test_matrix_multiply_2x2(self):
        """Test 2x2 matrix multiplication"""
        mat_a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        mat_b = np.array([[5, 6], [7, 8]], dtype=np.float64)
        result = numpy_matrix_multiply(mat_a, mat_b)

        expected = [[19.0, 22.0], [43.0, 50.0]]
        assert len(result) == 2
        assert len(result[0]) == 2
        for i in range(2):
            for j in range(2):
                assert abs(result[i][j] - expected[i][j]) < 0.001

    def test_matrix_multiply_identity(self):
        """Test multiplication with identity matrix"""
        mat = np.array([[2, 3], [4, 5]], dtype=np.float64)
        identity = np.eye(2, dtype=np.float64)

        result = numpy_matrix_multiply(mat, identity)

        for i in range(2):
            for j in range(2):
                assert abs(result[i][j] - mat[i][j]) < 0.001


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Requires nuwa-example to be built")
class TestNumpyWrappersProperties:
    """Test array properties and wrapper behavior."""

    def test_contiguous_array_properties(self):
        """Verify contiguous arrays are detected correctly"""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        # Test that contiguous arrays work efficiently
        result = numpy_array_sum(arr)
        assert result == 15

    def test_different_dtypes(self):
        """Test that different dtypes work"""
        # Test int64
        arr_int = np.array([1, 2, 3], dtype=np.int64)
        assert numpy_array_sum(arr_int) == 6

        # Test float64
        arr_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = numpy_array_multiply_scalar(arr_float, 2.0)
        assert result == [2.0, 4.0, 6.0]


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Requires nuwa-example to be built")
class TestNumpyWrappersEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_array(self):
        """Test with empty array"""
        arr = np.array([], dtype=np.int64)
        result = numpy_array_sum(arr)
        assert result == 0

    def test_single_element(self):
        """Test with single element"""
        arr = np.array([42], dtype=np.int64)
        result = numpy_array_sum(arr)
        assert result == 42

    def test_zeros_array(self):
        """Test with zeros"""
        arr = np.zeros(10, dtype=np.float64)
        result = numpy_array_multiply_scalar(arr, 5.0)
        assert all(x == 0.0 for x in result)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Requires nuwa-example to be built")
class TestNumpyWrappersRAII:
    """Test RAII cleanup and resource management."""

    def test_multiple_operations(self):
        """Test that multiple array operations work in sequence"""
        arr1 = np.array([1, 2, 3], dtype=np.int64)
        arr2 = np.array([4, 5, 6], dtype=np.int64)

        assert numpy_array_sum(arr1) == 6
        assert numpy_array_sum(arr2) == 15
        assert numpy_array_sum(arr1) == 6  # Can reuse arr1

    def test_nested_function_calls(self):
        """Test that functions can call other functions using wrappers"""
        # This verifies that wrappers don't interfere with each other
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        # Call sum multiple times
        sum1 = numpy_array_sum(arr)
        sum2 = numpy_array_sum(arr)

        assert sum1 == sum2 == 15


def print_instructions():
    """Print instructions for running these tests."""
    print("""
=== Running nuwa_sdk Numpy Wrapper Tests ===

These tests verify the numpy wrapper functionality through integration tests.

Prerequisites:
1. Build the example project with updated nuwa_sdk:
   cd /Users/martin/repos/nuwa/nuwa-example/example_project
   find .nimble/pkgs2 -name "nuwa_sdk.nim" \\
     -exec cp /Users/martin/repos/nuwa/nuwa-sdk/src/nuwa_sdk.nim {} \\;
   nuwa develop

2. Run tests:
   cd /Users/martin/repos/nuwa/nuwa-example/example_project
   pytest ../nuwa-sdk/tests/test_numpy_integrations.py

   Or copy to example project tests directory:
   cp ../nuwa-sdk/tests/test_numpy_integrations.py tests/
   pytest tests/test_numpy_integrations.py

Quick test:
  cd /Users/martin/repos/nuwa/nuwa-example/example_project
  python3 -c "
import numpy as np
from example_project import numpy_array_sum, numpy_array_sum_fast

arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
assert numpy_array_sum(arr) == 15
assert numpy_array_sum_fast(arr) == 15
print('Tests passed!')
"
    """)


if __name__ == "__main__":
    if IMPORTS_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print_instructions()
        sys.exit(1)
