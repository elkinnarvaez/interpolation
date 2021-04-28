import numpy as np
from gauss import gauss
from gauss import gauss_jordan
from back_substitution import back_substitution
from forward_substitution import forward_substitution
from helper import build_polynomial_matrix, build_newton_matrix

def interpolation_gauss(t, y):
    """
        Input: A set of pairs of integers (t[i], y[i]), 0 <= i < n, where each pair represents a data point
        Output: Solution of the linear system Ax = y, where A is the matrix that results from the evaluation of the training values in monomials
    """
    n = len(t)
    A = build_polynomial_matrix(n, n, t)
    b = np.array([[e] for e in y], dtype='float')
    return np.linalg.solve(A, b)

def interpolation_lagrange(t, y):
    """
        Input: A set of pairs of integers (t[i], y[i]), 0 <= i < n, where each pair represents a data point
        Description: Let Ax = y the system we want to solve. If we build the matrix A using the Lagrange function, it will correspond to the identity matrix. Therefore,
                     the solution to the system will be x = y, which is in fact the only thing that this function returns (this function might seem useless, but it is
                     very helpful for the sake of the well understanding of the lagrange interpolation method).
    """
    return np.array([[e] for e in y], dtype='float')

def interpolation_newton(t, y):
    """
        Input: A set of pairs of integers (t[i], y[i]), 0 <= i < n, where each pair represents a data point
        Output: Solution of the linear system Ax = y, where A is the matrix that results from the evaluation of the training values in the Newton interpolation function
    """
    n = len(t)
    A = build_newton_matrix(n, n, t, t)
    print(A.size)
    b = np.array([[e] for e in y], dtype='float')
    return forward_substitution(A, b)