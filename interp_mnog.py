import numpy as np
from numpy.polynomial.polynomial import Polynomial


def func(x):
    return np.abs(x)

def create_points(start_point, end_point, count_of_points, uniform_grid=True):

    if uniform_grid:
        points = np.linspace(start_point, end_point, count_of_points, dtype=np.float64)
    else:
        root = []
        for _ in range(count_of_points):
            root.append(0)
        root.append(1)
        points = np.polynomial.chebyshev.chebroots(root)
        points[0] = start_point
        points[-1] = end_point

    return points

def vand_interp(count_of_points, start_point, end_point, function, tolerance, uniform_grid=True):

    points = create_points(start_point, end_point, count_of_points, uniform_grid)

    vander = np.zeros((n, n))
    r_part = np.zeros(n)

    for i in range(n):
        for j in range(n):
            vander[i, j] =  points[i]**j
        r_part[i] = function(points[i])

    res = np.linalg.solve(vander, r_part)

    corr = True

    for i in range(n):
        val = 0
        for j in range(n):
            val += res[j] * points[i]**j
        if np.abs(val - r_part[i]) >= tolerance:
            corr = False

    if corr:
        print("ALL GOOD")
        print(res)
    else:
        print("Can't achieve that tolerance")

def mult(x, roots):

    val = 1

    for y in roots:
        val *= (x - y)

    return val

def lagrange_interp(count_of_points, start_point, end_point, function, tolerance, uniform_grid=True):

    points = create_points(start_point, end_point, count_of_points, uniform_grid)

    res = Polynomial([0])

    for i in range(count_of_points):

        roots = points.copy()

        roots = np.delete(roots, i)

        p = Polynomial.fromroots(roots)

        val = mult(points[i], roots)

        res += p * function(points[i]) / val

    n = res.degree() + 1

    corr = True

    for i in range(n):

        val = 0
        for j, x in enumerate(res):
            val += x * points[i]**j

        if np.abs(val - function(points[i])) >= tolerance:
            corr = False

    if corr:
        print("ALL GOOD")
        print(res)
    else:
        print("Can't achieve that tolerance")

    return res

def integral_1(val):
    if val % 2:
        return 0
    else:
        return 2 / (val + 1)
    
def polin_val(mas, val):
    res = 0
    for i, x in enumerate(mas):
        res += x * val**i
    return res

def ortog_polinom(count_of_points, start_point, end_point, function, tolerance, integral, uniform_grid=True):

    n = count_of_points

    G = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            G[i, j] = integral(i + j)

    L = np.linalg.inv(np.linalg.cholesky(G))

    l = []

    for i in range(n):
        l.append(Polynomial(L[i, :(i + 1)]))

    points = create_points(start_point, end_point, n, uniform_grid)

    r_part = np.zeros(n)

    for i in range(n):
        for j in range(n):
            G[i, j] = polin_val(l[j], points[i])
        r_part[i] = function(points[i])

    x_mas = np.linalg.solve(G, r_part)

    res = Polynomial([0])

    for i, x in enumerate(x_mas):
        res += x_mas[i] * l[i]

    corr = True

    for i in range(n):

        val = 0
        for j, x in enumerate(res):
            val += x * points[i]**j

        if np.abs(val - function(points[i])) >= tolerance:
            corr = False

    if corr:
        print("ALL GOOD")
        print(res)
    else:
        print("Can't achieve that tolerance")

        
    


deg = 2
a, b = -1, 1
eps = 10**(-10)
function = func
flag = "ortog_polinom"

n = deg + 1

if flag == "Vandermond":

    vand_interp(n, a, b, function, eps, False)

elif flag == "Lagrange":

    lagrange_interp(n, a, b, function, eps, True)

elif flag == "ortog_polinom":

    ortog_polinom(n, a, b, function, eps, integral_1, False)


