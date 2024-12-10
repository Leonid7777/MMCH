import numpy as np
from scipy.linalg import eigh_tridiagonal
from itertools import combinations
from functools import reduce
from operator import mul


def integral_1(val):
    if val % 2:
        return 0
    else:
        return 2 / (val + 1)
    
def dot_product(first_mas, second_mas, corr, integral):
    val = 0
    for ind_i, x in enumerate(first_mas):
        for ind_j, k in enumerate(second_mas):
            val += x * k * integral(ind_i + ind_j + corr)
    return val
    
def test(L, n, integral, tolerance):

    flag = True

    for i in range(0, n):
        val = dot_product(L[i, :(i + 1)], L[i, :(i + 1)], 0, integral)

        if (np.abs(np.abs(val) - 1) >= tolerance):
            flag = False
            break

        for j in range(i + 1, n):
            val = dot_product(L[i, :(i + 1)], L[j, :(j + 1)], 0, integral)

            if (np.abs(val) >= tolerance):
                flag = False

    if flag:
        print("ALL GOOD")
        print(L)
    else:
        print("Can't achive that tolerance")


    
max_deg = 10
eps = 10**(-5)
integral = integral_1
method = "Eigenvalues"

n = max_deg + 1

integral = integral_1

if method == 'Gramm':

    G = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            G[i, j] = integral(i + j)

    L = np.linalg.inv(np.linalg.cholesky(G))

    test(L, n, integral, eps)

elif method == "Recur":

    L = np.zeros((n, n))

    L[0, 0] = np.sqrt(0.5)

    betta = 0
    
    for i in range(1, n):

        alpha = dot_product(L[i - 1], L[i - 1], 1, integral)

        L[i, 1:(i + 1)] = L[(i - 1), :i] 
        L[i, :i] -= alpha * L[i - 1, :i]

        if betta:
            L[i, :(i - 1)] -= betta * L[i - 2, :(i - 1)]

        betta = np.sqrt(dot_product(L[i], L[i], 0, integral))

        L[i, :(i + 1)] /= betta

    test(L, n, integral, eps)

elif method == "Eigenvalues":

    L = np.zeros((n, n))

    L[0, 0] = np.sqrt(0.5)

    betta = 0

    mas_alpha = []
    mas_betta = []

    for i in range(1, n):

        alpha = dot_product(L[i - 1], L[i - 1], 1, integral)

        mas_alpha.append(alpha)

        roots = eigh_tridiagonal(np.array(mas_alpha), np.array(mas_betta))[0]

        for j in range(i):
            val = 0
            for x in [*combinations(roots, i - j)]:
                val += reduce(mul, x)
            L[i, j] = (-1)**(i - j) * val
        L[i, i] = 1

        norm = np.sqrt(dot_product(L[i], L[i], 0, integral))

        L[i, :(i + 1)] /= norm

        betta = dot_product(L[i], L[i - 1], 1, integral)

        mas_betta.append(betta)

    test(L, n, integral, eps)
