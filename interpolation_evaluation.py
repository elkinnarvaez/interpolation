import  numpy as np
import pandas as pd
from helper import build_polynomial_matrix, build_lagrange_matrix, build_newton_matrix

def polynomial_evaluation(t, x):
    n = len(x)
    m = len(t)
    A = build_polynomial_matrix(n, m, t)
    b = np.matmul(A, x)
    return b

def lagrange_evaluation(t, t_train, x):
    n = len(x)
    m = len(t)
    A = build_lagrange_matrix(n, m, t, t_train)
    b = np.matmul(A, x)
    return b

def newton_evaluation(t, t_train, x):
    n = len(x)
    m = len(t)
    A = build_newton_matrix(n, m, t, t_train)
    b = np.matmul(A, x)
    return b