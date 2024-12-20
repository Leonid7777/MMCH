import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc

def func(x):
    # return 5 * x + 7
    return np.exp(x)

def test(start_point, end_point, function, tolerance, val):
    err = np.abs(quad(function, start_point, end_point)[0] - val)
    if err <= tolerance:
        print("ALL GOOD")
        print(f"Error = {err}")
        print(f"Integral = {val}")
    else:
        print("Can't achieve that tolerance")
        print(f"Error = {err}")
        print(f"Integral = {val}")

def monte_carlo(start_point, end_point, tolerance, function):

    count_of_point = 1000
    length = end_point - start_point

    first_val = 0
    second_val = 10 * tolerance 

    while np.abs(first_val - second_val) >= tolerance / 2:
        points = start_point + np.random.rand(count_of_point) * length

        second_val = first_val
        first_val = 0

        for i in range(count_of_point):
            first_val += function(points[i])

        first_val *= length / count_of_point

        count_of_point *= 10

    test(start_point, end_point, function, tolerance, first_val)

def geomic_monte_carlo(start_point, end_point, tolerance, function, max_val, grid, sigma):

    length = end_point - start_point

    volume = length * max_val

    first_val = 0

    count_of_point = int(np.ceil( (sigma * length / tolerance)**2 / 12) )

    print(count_of_point)

    if grid == "Uniform":
        points_x = start_point + np.random.rand(count_of_point) * length
        points_y = np.random.rand(count_of_point) * max_val
    elif grid == "Sobol":
        sampler = qmc.Sobol(d=2, scramble=False)
        points = sampler.random(count_of_point)
        points_x = [x[0] * length + start_point for x in points]
        points_y = [x[1] * max_val for x in points]
    else:
        sampler = qmc.Sobol(d=2, scramble=True)
        points = sampler.random(count_of_point)
        points_x = [x[0] * length + start_point for x in points]
        points_y = [x[1] * max_val for x in points]

    first_val = 0

    for i in range(count_of_point):
        first_val += (function(points_x[i]) >= points_y[i])

    first_val *= volume / count_of_point

    test(start_point, end_point, function, tolerance, first_val)

start_point = 0.1
end_point = 2
tolerance = 5 * 10**(-2)
max_val = np.exp(3) 
sigma = 5
method = "geomic_monte_carlo"
grid = "Sobol"

if method == "monte_carlo":
    monte_carlo(start_point, end_point, tolerance, func)
elif method == "geomic_monte_carlo":
    geomic_monte_carlo(start_point, end_point, tolerance, func, max_val, grid, sigma)
