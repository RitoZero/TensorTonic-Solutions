import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asanyarray(A)
    if A.ndim < 2:
        return A
    old_rows, old_cols = A.shape
    result = np.empty((old_cols, old_rows), dtype=A.dtype)

    for i in range(old_rows):
        for j in range(old_cols):
            result[j, i] = A[i, j]

    return result
