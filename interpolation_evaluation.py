import  numpy as np
import pandas as pd
from helper import build_monomial_matrix, build_lagrange_matrix, build_newton_matrix

def monomial_evaluation(t, x):
    n = len(x)
    m = len(t)
    A = build_monomial_matrix(n, m, t)
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

def piecewise_evaluation(t, t_train, X):
    n = len(t_train)
    m = len(t)
    b = [None for _ in range(m)]
    for i in range(m):
        j = 0
        # Find the place where t[i] fits
        while(j < n - 1):
            if(t[i] >= t_train[j] and t[i] <= t_train[j + 1]):
                break
            j += 1
        b[i] = X[j][0] + X[j][1]*t[i]
    return b