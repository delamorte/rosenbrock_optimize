import numpy as np
from scipy.optimize import minimize
"""
Example of minimizing the Rosenbrock function with different
methods using SciPy.

"""


def rosenbrock(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


# Randomize initial starting points for N-dimensions
# when  1.0 <= x <= 10.0
N = 9
x0 = np.random.uniform(low=1.0, high=10.0, size=N)
print(x0)

# Nelder-Mead method
print('\nResults of minimizing with Nelder-Mead method: ')
res_NM = minimize(rosenbrock, x0, method='Nelder-Mead',
                  options={'disp': True, 'maxiter': 10**4})
print('Solution: ')
print(res_NM.x)

# Sequential Least SQuares Programming (SLSQP) method
print('\nResults of minimizing with SLSQP method: ')
res_SLSQP = minimize(rosenbrock, x0, method='SLSQP',
                     options={'disp': True, 'maxiter': 10**4})
print('Solution: ')
print(res_SLSQP.x)

# Powell method
print('\nResults of minimizing with Powell method: ')
res_Powell = minimize(rosenbrock, x0, method='Powell',
                      options={'disp': True, 'maxiter': 10**4})
print('Solution: ')
print(res_Powell.x)
