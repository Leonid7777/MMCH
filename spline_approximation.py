import numpy as np


def F(x):
    return np.sin(4*x)

def run_through_method(side_diag, main_diag, r_part):

    n = main_diag.shape[0]

    x = np.zeros(n)
    alpha = np.zeros(n - 1)
    betta = np.zeros(n - 1)

    alpha[0] = - side_diag[0] / main_diag[0]
    betta[0] = r_part[0] / main_diag[0]

    for i in range(1, n - 1):
        devider = side_diag[i] * alpha[i - 1] + main_diag[i]
        alpha[i] = - side_diag[i] / devider
        betta[i] = (r_part[i] - side_diag[i] * betta[i - 1]) / devider

    x[-1] = (r_part[-1] - side_diag[-1] * betta[-1]) / (side_diag[-1] * alpha[-1] + main_diag[-1])

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + betta[i]

    return x

def function_delta(f_point, s_point, t_point, f_segment, s_segment, function):
    f_val =  (function(f_point) - function(s_point)) / f_segment
    s_val = (function(s_point) - function(t_point)) / s_segment
    return 6 * (f_val - s_val)

def create_r_part(points, segments, function):
    vals = []

    for i in range(segments.shape[0] - 1):
        new_val = function_delta(points[i + 2], points[i + 1], points[i], segments[i + 1], segments[i], function)
        vals.append(new_val)

    return np.array(vals)

def spline_approx(start_point, end_point, function, count_of_points, uniform_grid=True):

    if uniform_grid:
        points = np.linspace(start_point, end_point, count_of_points, dtype=np.float64)
    else:
        points = np.sort(np.random.rand(count_of_points)) * (end_point - start_point) + start_point
        points[0] = start_point
        points[-1] = end_point

    segments = points[1:] - points[:-1]

    F_points =  create_r_part(points, segments, function)

    main_diag = 2 * (segments[:-1] + segments[1:])

    side_diag = segments[1:-1]

    res = run_through_method(side_diag, main_diag, F_points)

    res = np.append(res, 0)
    res = np.insert(res, 0, 0)

    polin = []
    for i in range(1, count_of_points):

        func_point = function(points[i - 1])
        deriv = (function(points[i]) - func_point) / segments[i - 1] - (res[i] + 2 * res[i - 1]) * segments[i - 1] /  6 
        diff = (res[i] - res[i - 1]) / (2 * segments[i - 1])
        
        a_3 = (res[i] - res[i - 1]) / (6 * segments[i - 1])
        a_2 = res[i - 1] / 2 - diff * points[i - 1]
        a_1 = deriv - points[i - 1] * res[i - 1] + diff * points[i - 1]**2 
        a_0 = func_point - points[i - 1] * deriv + res[i - 1] * points[i - 1]**2 / 2 - diff * points[i - 1]**3 / 3
        polin.append([a_3, a_2, a_1, a_0])

    return np.array(polin)


np.random.seed(42)
a, b = -1, 1
n = 9
res = spline_approx(start_point=a, end_point=b, function=F, count_of_points=n, uniform_grid=True)
print(res)
