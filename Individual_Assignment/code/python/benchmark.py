from matrix_multiplication import multiply_matrixes
import numpy as np
import pytest

@pytest.mark.parametrize("dim", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
def test_my_function(benchmark, dim):
    dim1 = dim2 = dim3 = dim
    A = np.random.rand(dim1, dim2)
    B = np.random.rand(dim2, dim3)
    C = np.zeros((dim1, dim3))

    result = benchmark(multiply_matrixes, A, B, C, dim1, dim2, dim3)
    assert np.allclose(A.dot(B), C)
