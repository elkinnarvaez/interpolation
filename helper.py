import  numpy as np
import pandas as pd

def build_polynomial_matrix(n, m, t):
    A = np.array([[None for _ in range(n)] for _ in range(m)], dtype='float')
    for i in range(m):
        for j in range(n):
            A[i][j] = t[i]**j
    return A

def l(j, ti, t_train, n):
    numerator = 1
    denominator = 1
    for k in range(n):
        if(k != j):
            numerator = numerator * (ti - t_train[k])
    for k in range(n):
        if(k != j):
            denominator = denominator * (t_train[j] - t_train[k])
    return round(numerator/denominator, 2)

def build_lagrange_matrix(n, m, t, t_train):
    A = np.array([[None for _ in range(n)] for _ in range(m)], dtype='float')
    for i in range(m):
        for j in range(n):
            A[i][j] = l(j, t[i], t_train, len(t_train))
    return A

def newton(j, ti, t_train):
    ans = 1
    for k in range(j):
        ans = ans * (ti - t_train[k])
    return ans

def build_newton_matrix(n, m, t, t_train):
    A = np.array([[None for _ in range(n)] for _ in range(m)], dtype='float')
    for i in range(m):
        for j in range(n):
            A[i][j] = newton(j, t[i], t_train)
    return A