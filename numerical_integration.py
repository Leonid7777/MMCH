import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad
import scipy.special


def func(x):
    return np.exp(x)

def replace_point(start_point, end_point, point):
    return ((start_point + end_point) + (end_point - start_point) * point) / 2

def mult(x, roots):
    val = 1
    for y in roots:
        val *= (x - y)
    return val

def test(val, function, start_point, end_point, tolerance):
    result = quad(function, start_point, end_point)

    diff = np.abs(result[0] - val)

    if diff < tolerance:
        print("ALL GOOD") 
    else:
        print("Can't achieve that tolerance")

    print(f'dif={diff}')
    print(f'result={result[0]}')
    print(f'val={val}') 

def newton_cotes(start_point, end_point, count_of_points, function, tolerance):

    points_t = np.linspace(-1, 1, count_of_points + 1, dtype=np.float64)
    f_part = np.array([function(replace_point(start_point, end_point, point)) for point in points_t])

    val = 0

    for i in range(count_of_points + 1):

        roots = points_t.copy()
        roots = np.delete(roots, i)

        p = (Polynomial.fromroots(roots)).integ()

        val += f_part[i] * (p(1) - p(-1)) / mult(points_t[i], roots)

    val *= (end_point - start_point) / 2

    test(val, function, start_point, end_point, tolerance)

def gaus(start_point, end_point, count_of_points, function, tolerance):

    points_t = scipy.special.p_roots(count_of_points)[0]
    f_part = np.array([function(replace_point(start_point, end_point, point)) for point in points_t])

    val = 0

    for i in range(count_of_points):

        pol_der = (scipy.special.lpn(count_of_points, points_t[i])[1][-1])**2

        coef = 2 / ((1 - points_t[i]**2) * pol_der)

        val += coef * f_part[i]

    val *= (end_point - start_point) / 2

    test(val, function, start_point, end_point, tolerance)

def clenshaw_curtis(start_point, end_point, count_of_points, function, tolerance):

    maxim = count_of_points // 2
    znam = [1 - 4*k**2 for k in range(maxim)]
    val = 0

    for j in range(count_of_points + 1):
        w = 1 / 2
        mult = 2 * j * np.pi / count_of_points
        ch = 0
        for k in range(1, maxim):
            ch += mult
            w += np.cos(ch) / znam[k]
        
        w *= 4 / count_of_points

        if not j or j == count_of_points:
            w /= 2
        
        val += w * function(replace_point(start_point, end_point, np.cos(mult / 2)))

    val *= (end_point - start_point) / 2

    test(val, function, start_point, end_point, tolerance)


count_of_points = 30
a, b = -10, 10
eps = 10**(-5)
method = "clenshaw_curtis"

if method == "newton_cotes":
    newton_cotes(a, b, count_of_points, func, eps)
elif method == "gaus":
    gaus(a, b, count_of_points, func, eps)
elif method == "clenshaw_curtis":
    clenshaw_curtis(a, b, count_of_points, func, eps)
