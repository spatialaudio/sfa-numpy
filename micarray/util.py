import numpy as np
from scipy import linalg


def norm_of_columns(A, p=2):
    """Vector p-norm of each column."""
    _, N = A.shape
    return np.asarray([linalg.norm(A[:, j], ord=p) for j in range(N)])


def coherence_of_columns(A):
    """Mutual coherence of columns of A."""
    A = np.asmatrix(A)
    _, N = A.shape
    A = A * np.asmatrix(np.diag(1/norm_of_columns(A)))
    Gram_A = A.H*A
    for j in range(N):
        Gram_A[j, j] = 0
    return np.max(np.abs(Gram_A))
