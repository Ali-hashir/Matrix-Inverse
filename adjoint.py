# def matrix_multiply(matrix1, matrix2):
#     result = []
#     for i in range(len(matrix1)):
#         row = []
#         for j in range(len(matrix2[0])):
#             element = 0
#             for k in range(len(matrix2)):
#                 element += matrix1[i][k] * matrix2[k][j]
#             row.append(element)
#         result.append(row)
#     return result
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

# Example usage for a 3x3 matrix
# matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
# inverse = adjoint_inverse(matrix)

# print("Inverse matrix:")
# for row in inverse:
#     print(row)