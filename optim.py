import numpy as np
from scipy.optimize import minimize

def objective(x, *f):
    if not isinstance(x, list):
        x = list(x)
    returnValue = 0
    f = list(*f)
    for i in range(len(x)):
        returnValue -= f[i](x[i])*x[i]/sum(x)
    return returnValue

def value_constraint(x, value):
    return -sum(x) + value

def multistart_minimize(objective, interpolationFunction: list, bnds: list, cons, n_functions, n_trials=100):
    """Multi-start local optimization of a scalar function."""
    best = None
    rng = np.random.default_rng()
    points = np.round(np.max(bnds)*rng.random((n_trials,n_functions)))
    # now = {"success": False}
    for i in range(n_trials):
        try:
            now = minimize(objective, x0 = points[i,:], args = (interpolationFunction), bounds = bnds, constraints=cons)
            # now = minimize(objective, x0 = points[i,:], args = (interpolationFunction), bounds = bnds, constraints=cons, method = "trust-constr")
            if now.success and (best is None or best.fun > now.fun):
                best = now
        except ValueError:
            print("Value Error")
            # pass
    return best