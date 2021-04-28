import numpy as np

def compute_sum(A, b, x, i, n):
    numerator = b[i][0]
    s = 0
    for j in range(i + 1, n):
        s += A[i][j]*x[j][0]
    numerator = numerator - s
    denominator = A[i][i]
    return numerator/denominator

def back_substitution(A, b):
    n = b.size
    x = np.array([[None] for _ in range(n)], dtype='float')
    for i in range(n - 1, -1, -1):
        x[i][0] = round(compute_sum(A, b, x, i, n), 4)
    return x