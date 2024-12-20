def func(x):
    # return 3 * x**5 - 4 * x**4 + 10 * x**3 - 5 * x**2 + 5*x - 10
    return 5 * x**2 + 10 * x + 7

def func_der(x):
    # return 15 * x**4 - 16 * x**3 + 30 * x**2 - 10 * x + 5
    return 10 * x + 10 

def test(x, function, tolerance):
    if abs(function(x)) < tolerance:
        print("ALL GOOD")
        print(function(x))
    else:
        print("Can't achieve that tolerance")

def newton(start_point, function, function_deriv, tolerance):
    x = start_point

    while abs(function(x) / function_deriv(x)) >= tolerance:
        x = x - function(x) / function_deriv(x)

    test(x, function, tolerance)

def gd(start_point, function, function_deriv, tolerance, alpha):
    x = start_point

    first = func(x)
    second = 10 * first

    while abs(first - second) >= tolerance:
        second = first
        x = x - alpha * function_deriv(x)
        first = function(x)
        
    print(function(x))


start_point = 10
eps = 10**(-5)
method = "GD"
if method == "Newton":
    newton(start_point, func, func_der, eps)
elif method == "GD":
    alpha = 0.01
    gd(start_point, func, func_der, eps, alpha)
