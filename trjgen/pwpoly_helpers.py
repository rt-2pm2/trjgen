"""
Helper functions for Piecewise polynomial


@author: @author: l.pannocchi@gmail.com
"""

import numpy as np

## =================================================
## =================================================
def pwpTo1D(T, pwpoly, der):
    """
    Generate the 1D trajectory in space from polynomial

    Args
        T:      Vector of points where the polynomial is evaluated
        pwpoly: Piecewise polynomial object
        der:    Order of the derivatives to be evaluated

    Outputs
        X:      Vector/Matrix with the evaluated polynomial
                NumDerivative x NumberOfPoints
    """

    # Check whether the input is a scalar
    if type(der) == int:
        der = [der]

    der = np.array(der)
    T = np.array(T)

    X = np.zeros((der.size, T.size))

    for i in range(der.size):
        X[i,:] = pwpoly.eval(T, der[i])

    return X