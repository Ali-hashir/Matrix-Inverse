# ---------------------------------------Brute Force---------------------------------------
import copy

def minor(matrix, row, column):
    minor = copy.deepcopy(matrix)
    minor = minor[:row] + minor[row+1:]
    for i in range(len(minor)):
        minor[i] = minor[i][:column] + minor[i][column+1:]
    return minor

def determinant(matrix):
    n = len(matrix)
    
    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    
    # Base case for 1x1 matrix
    elif n == 1:
        return matrix[0][0]

    det = 0
    for c in range(n):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]] # Get minor of element (0, c)
        det += ((-1) ** c) * matrix[0][c] * determinant(minor) # Recursive call

    return det


def cofactor(matrix):
    cofactors = []
    for r in range(len(matrix)):
        cofactorRow = []
        for c in range(len(matrix)):
            minor_det = determinant(minor(matrix, r, c))
            cofactorRow.append(((-1) ** (r + c)) * minor_det)
        cofactors.append(cofactorRow)
    return cofactors

def transpose(matrix):
    transposed = []
    for c in range(len(matrix)):
        transposedRow = []
        for r in range(len(matrix)):
            transposedRow.append(matrix[r][c])
        transposed.append(transposedRow)
    return transposed

def scalar_multiplication(matrix, scalar):
    for r in range(len(matrix)):
        for c in range(len(matrix)):
            matrix[r][c] *= scalar
    return matrix

def matrix_inverse(matrix):
    det = determinant(matrix)
    if det == 0:
        return "Matrix is not invertible"
    else:
        cofactors = cofactor(matrix)
        adjugate = transpose(cofactors)
        inverse = scalar_multiplication(adjugate, 1/det)
        return inverse

# ---------------------------------------Gauss Jordan---------------------------------------

def gauss_jordan(matrix):
    n = len(matrix)

    # Step 1: Create an augmented matrix [A|I]
    identity = [[float(i ==j) for i in range(n)] for j in range(n)]
    for i in range(n):
        matrix[i] += identity[i]

    # Step 2: Perform Gauss Jordan Elimination to transform A into RREF
    for i in range(n):
        if matrix[i][i] == 0:
            for j in range(i+1, n):
                if matrix[j][i] != 0:
                    matrix[i], matrix[j] = matrix[j], matrix[i]
                    break
            else:
                return "Matrix is not invertible"
        
        pivot = matrix[i][i]
        for j in range(i, 2*n):
            matrix[i][j] /= pivot

        for j in range(n):
            if i != j:
                ratio = matrix[j][i]
                for k in range(2*n):
                    matrix[j][k] -= ratio * matrix[i][k]

    # Step 3: If A is now an identity matrix, then I is the inverse matrix
    # Extract the inverse matrix from [I|A_inv]
    inverse_matrix = [row[n:] for row in matrix]
    
    return inverse_matrix

# ---------------------------------------adjoint---------------------------------------
def matrix_multiply(M, N):
    """Multiply square matrices of same dimension M and N"""
    # Converts N into a list of tuples of columns
    tuple_N = list(zip(*N))
    # Nested list comprehension to calculate matrix multiplication
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]

def matrix_transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Calculate the determinant
def determinant(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i in range(len(matrix)):
        sub_matrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += (-1) ** i * matrix[0][i] * determinant(sub_matrix)
    return det

 # Calculate the adjugate matrix
def adjugate(matrix):
    cofactors = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix[0])):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            cofactor = (-1) ** (i + j) * determinant(sub_matrix)
            cofactor_row.append(cofactor)
        cofactors.append(cofactor_row)
    return matrix_transpose(cofactors)
    
def adjoint_inverse(matrix):
    n = len(matrix)

    det = determinant(matrix)
    if det == 0:
        raise ValueError("Matrix is not invertible.")

    adj_matrix = adjugate(matrix)

    # Multiply adjugate matrix by the determinant inverse
    det_inv = 1 / det
    inv_matrix = [[det_inv * adj_matrix[i][j] for j in range(n)] for i in range(n)]
    return inv_matrix

# ---------------------------------------LUD---------------------------------------

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
    """Multiply square matrices of same dimension M and N"""
    # Converts N into a list of tuples of columns
    tuple_N = list(zip(*N))
    # Nested list comprehension to calculate matrix multiplication
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]

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

# ---------------------------------------Test Cases---------------------------------------
import time
import matplotlib.pyplot as plt
import random

# Sizes of the matrices
sizes = [7, 8, 9, 10]

# Store execution times
brute_force_times = []
gauss_jordan_times = []
adjoint_times = []
LUD_times = []

for n in sizes:
    # Generate a random nxn matrix
    matrix = [[random.random() for _ in range(n)] for _ in range(n)]

    # Time the brute force method
    start_time = time.time()
    matrix_inverse(matrix)
    end_time = time.time()
    brute_force_times.append(end_time - start_time)

    # Time the Gauss Jordan method
    start_time = time.time()
    gauss_jordan(matrix)
    end_time = time.time()
    gauss_jordan_times.append(end_time - start_time)

    # Time the adjoint method
    start_time = time.time()
    adjoint_inverse(matrix)
    end_time = time.time()
    adjoint_times.append(end_time - start_time)

    # Time the LUD method
    start_time = time.time()
    LUD_inverse(matrix)
    end_time = time.time()
    LUD_times.append(end_time - start_time)



# Create a new figure
plt.figure()

# Plot the execution times
plt.plot(sizes, brute_force_times, label='Brute Force')
plt.plot(sizes, gauss_jordan_times, label='Gauss Jordan')

# Add labels and legend
plt.xlabel('Size of the matrix (n)')
plt.ylabel('Execution time (seconds)')
plt.legend()

# Show the plot
plt.show()
