import numpy as np
from scipy.optimize import newton


def F(x):
    return np.exp(x)

def p(x, mas):
    val = 0
    for i in range(mas.shape[0]):
        val += mas[i] * x**i
    return val

def maximize_scan(p_coef, a, b, num_points=1000):
    x = np.linspace(a, b, num_points)
    y = np.array([np.abs(F(val) - p(val, p_coef)) for val in x])
    max_index = np.argmax(y)
    return x[max_index]


eps = 10**(-2)
deg = 3
a, b = -1, 1

points = np.sort(np.random.rand(deg + 2) * (b - a) + a)
vander = np.column_stack((np.vander(points, deg + 1), np.array([(-1)**i for i in range(deg + 2)])))
F_val = np.array([F(x) for x in points])

err = eps * 100

while err > eps:

    res = np.linalg.solve(vander, F_val)

    p_coef = res[-2::-1]
    d = res[-1]

    new_point = maximize_scan(p_coef, a, b)

    func_res = F(new_point) - p(new_point, p_coef)

    ind = np.argwhere(points > new_point)
    if ind.shape[0]:
        ind = ind.min()
        if ind == 0 and vander[ind, -1] * d * func_res < 0:
            vander[1:, :] = vander[:-1, :]
            F_val[1:] = F_val[:-1]
            vander[ind, -1] = -vander[ind, -1]
        elif vander[ind, -1] * d * func_res < 0:
                ind -= 1
    else:
        ind = deg + 1
        if vander[ind, -1] * d * func_res < 0:
            vander[:-1, :] = vander[1:, :]
            F_val[:-1] = F_val[1:]
            vander[ind, -1] = -vander[ind, -1]

    vander[ind, :-1] = np.array([new_point**i for i in range(deg, -1, -1)])
    F_val[ind] = F(new_point)

    err = np.abs(np.abs(func_res) - np.abs(d))

    print(f'err = {err}')
    print(f'func_res = {func_res}')
    print()


print(f'coef of polinom = {p_coef}')
