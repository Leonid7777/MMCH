import numpy as np
from numpy.polynomial.polynomial import Polynomial


def func(x, step):
    if not step % 4:
        return np.sin(x)
    elif step % 4 == 1:
        return np.cos(x)
    elif step % 4 == 2:
        return -np.sin(x)
    else:
        return -np.cos(x)


def create_points(start_point, end_point, count_of_points, uniform_grid=True):

    n = int((count_of_points + 1) / 2)

    if uniform_grid:
        points = np.linspace(start_point, end_point, n, dtype=np.float64)
    else:
        root = []
        for _ in range(n):
            root.append(0)
        root.append(1)
        points = np.polynomial.chebyshev.chebroots(root)
        points[0] = start_point
        points[-1] = end_point

    points_left = count_of_points - n

    randomized = np.random.choice(n, points_left)

    points = np.sort(np.append(points, points[randomized]))

    return points

def multip_newton_interp(start_point, end_point, count_of_points, function, tolerance, uniform_grid=True):

    n = count_of_points

    points = create_points(start_point, end_point, n, uniform_grid)

    l = [[function(x, 0) for x in points]]

    for i in range(1, n):
        div = []
        for j in range(0, count_of_points - i):
            if np.abs(points[j] - points[j + i]) < 10**(-10):
                div.append(function(points[j], i) / np.math.factorial(i))
            else:
                div.append((l[i - 1][j] - l[i - 1][j + 1]) / (points[j] - points[j + i]))
        l.append(div)

    res = Polynomial([l[0][0]])

    for i in range(1, n):
        res += l[i][0] * Polynomial.fromroots(points[: i])


    unique, counts = np.unique(points, return_counts=True)

    corr = True

    for i in range(unique.shape[0]):

        for k in range(counts[i]):        
            val = 0
            diverg = res.deriv(k)
            for j, x in enumerate(diverg):
                val += x * unique[i]**j

            if np.abs(val - function(unique[i], k)) >= tolerance:
                corr = False

    if corr:
        print("ALL GOOD")
        print(res)
    else:
        print("Can't achieve that tolerance")
        print(res)

    return res


# np.random.seed(42)
deg = 30
a, b = -1, 1
eps = 10**(-2)
function = func

n = deg + 1

multip_newton_interp(a, b, n, func, eps, True)
