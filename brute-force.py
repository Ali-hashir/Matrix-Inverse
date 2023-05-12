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


def main():
    matrix = [[1, 0], [0, 1]]
    print(matrix_inverse(matrix))

if __name__ == "__main__":
    main()