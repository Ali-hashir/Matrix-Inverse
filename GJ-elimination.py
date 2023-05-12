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

import time

def main():

    matrix = [[5, 7, 3], [5, 6, 7], [8, 5, 3]]
    start = time.time()
    print(gauss_jordan(matrix))
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()