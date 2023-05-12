import numpy as np
import timeit
import matplotlib.pyplot as plt
from LUD import *
from BruteForce import *
from adjoint import *
from GJE import *
# Function to generate a random square matrix
import random

def generate_matrix(size):
    matrix = [[random.randint(-10, 10) for _ in range(size)] for _ in range(size)]
    return matrix

# Perform benchmarking and generate performance graphs
def compare_methods(matrix_sizes, num_iterations):
    gauss_jordan_times = []
    adjoint_times = []
    lu_times = []
    nplu_times = []
    brute_times = []
    

    for size in matrix_sizes:
        gauss_jordan_avg_time = timeit.timeit(lambda: gauss_jordan(generate_matrix(size)), number=num_iterations)
        lu_avg_time = timeit.timeit(lambda: LUD_inverse(generate_matrix(size)), number=num_iterations)
        nplu_avg_time = timeit.timeit(lambda: LUD_np_inverse(generate_matrix(size)), number=num_iterations)
        brute_avg_time = timeit.timeit(lambda: BruteForce_inverse(generate_matrix(size)), number=num_iterations)
        adjoint_avg_time = timeit.timeit(lambda: adjoint_inverse(generate_matrix(size)), number=num_iterations)

        gauss_jordan_times.append(gauss_jordan_avg_time)
        lu_times.append(lu_avg_time)
        nplu_times.append(nplu_avg_time)
        brute_times.append(brute_avg_time)
        adjoint_times.append(adjoint_avg_time)


    # Plotting combined graph
    plt.plot(matrix_sizes, gauss_jordan_times, label="Guass Jordan Method")
    plt.plot(matrix_sizes, adjoint_times, label="Adjoint Method")
    plt.plot(matrix_sizes, lu_times, label="LU Decomposition Method")
    plt.plot(matrix_sizes, nplu_times, label="Scipy LU Decomposition")
    plt.plot(matrix_sizes, brute_times, label="Brute Force Metnod")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (seconds)")
    plt.title("Inverse Matrix Computation Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("combined_graph.png")
    plt.show()

# Set the matrix sizes and number of iterations for benchmarking
matrix_sizes = [3,4,5]
num_iterations = 5

# Perform benchmarking and generate graphs
compare_methods(matrix_sizes, num_iterations)
