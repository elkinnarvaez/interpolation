from back_substitution import back_substitution
from forward_substitution import forward_substitution
import numpy as np

def diagonal_division(A, b):
    n = b.size
    x = np.array([[None] for _ in range(n)], dtype='float')
    for i in range(n):
        x[i][0] = round(b[i][0]/A[i][i], 4)
    return x

def gauss(A, b):
    n = b.size
    if(np.linalg.det(A) != 0):
        for j in range(n - 1):
            M = np.identity(n)
            P = np.identity(n)
            non_zero = False
            if(A[j][j] == 0):
                i = 0
                while(i < n and not non_zero):
                    if(A[i][j] != 0):
                        P[j][j] = 0; P[i][j] = 1; P[i][i] = 0; P[j][i] = 1
                        non_zero = True
                    i += 1
            if(A[j][j] != 0 or non_zero):
                A = np.matmul(P, A)
                b = np.matmul(P, b)
                for i in range(j + 1, n):
                    M[i][j] = -1*(A[i][j]/A[j][j])
                A = np.matmul(M, A)
                b = np.matmul(M, b)
            else:
                raise NameError("Singular matrix")
        return back_substitution(A, b)
    else:
        raise NameError("Singular matrix")

def gauss_jordan(A, b):
    n = b.size
    if(np.linalg.det(A) != 0):
        for j in range(n):
            M = np.identity(n)
            P = np.identity(n)
            non_zero = False
            if(A[j][j] == 0):
                i = 0
                while(i < n and not non_zero):
                    if(A[i][j] != 0):
                        P[j][j] = 0; P[i][j] = 1; P[i][i] = 0; P[j][i] = 1
                        non_zero = True
                    i += 1
            if(A[j][j] != 0 or non_zero):
                A = np.matmul(P, A)
                b = np.matmul(P, b)
                for i in range(0, n):
                    M[i][j] = -1*(A[i][j]/A[j][j])
                A = np.matmul(M, A)
                b = np.matmul(M, b)
            else:
                raise NameError("Singular matrix")
        return diagonal_division(A, b)
    else:
        raise NameError("Singular matrix")