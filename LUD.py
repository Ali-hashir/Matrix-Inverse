from scipy.linalg import lu
import numpy as np
import pprint

def LUD_np_inverse(A):
    p,l,u = lu(A, permute_l = False)

    l = np.dot(p,l) 
    l_inv = np.linalg.inv(l)
    u_inv = np.linalg.inv(u)
    A_inv = np.dot(u_inv,l_inv)
    return A_inv


def mult_matrix(M, N):
    """Multiply square matrices of same dimension M and N using dynamic programming"""
    # Get the dimensions of the matrices
    m = len(M)
    n = len(N)
    p = len(N[0])

    # Create a result matrix of appropriate size filled with zeros
    result = [[0] * p for _ in range(m)]

    # Perform matrix multiplication using dynamic programming
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += M[i][k] * N[k][j]

    return result


def pivot_matrix(M):
    """Returns the pivoting matrix for M, used in Doolittle's method."""
    m = len(M)
    # Create an identity matrix, with floating point values
    id_mat = [[float(i == j) for i in range(m)] for j in range(m)]
    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of M
    for j in range(m):
        row = max(range(j, m), key=lambda i: abs(M[i][j]))
        if j != row:
            # Swap the rows
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]
    return id_mat

def lu_decomposition(A):
    n = len(A)
    # Create zero matrices for L and U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    # Create the pivot matrix P and the multiplied matrix PA
    P = pivot_matrix(A)
    PA = mult_matrix(P, A)
    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j][j] = 1.0
        # LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j + 1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1
        # LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]
    return P, L, U
# code taken from https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy/
def forward_substitution(L, b):
    n = len(L)
    y = [0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

def backward_substitution(U, y):
    n = len(U)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def LUD_inverse(matrix):
    P, L, U = lu_decomposition(matrix)
    n = len(matrix)
    identity = [[float(i == j) for j in range(n)] for i in range(n)]
    inverse = []
    for col in identity:
        y = forward_substitution(L, col)
        x = backward_substitution(U, y)
        inverse.append(x)

    inverse = mult_matrix(transpose(P), inverse)
    return inverse

A = [[1, 2, 3], [4, 5,6],[7, 8, 10]]
# pprint.pprint(inverse_using_lu_decomposition(A))
