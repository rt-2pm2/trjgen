"""
Bezier Polynomial
@author: l.pannocchi@gmail.com
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import trjgen.trjgen_core as trj_core


## =================================================
## =================================================
# Generate the Pascal Matrix
def genPascal(N):
    """
    Generate the Pascal Matrix containing
    the coefficient of the binomial expansion
    """
    P = np.eye(N, dtype=float)
    P[:, 0] = 1.0
    for i in range(2, N):
        for j in range(1, i):
            P[i,j] = P[i-1, j-1] + P[i-1, j]
    return P

def genBezierM(deg):
    N = deg + 1
    PascalM = genPascal(N)
    BM = np.zeros((N,N), dtype=float)
    for j in range(N):
        for i in range(j, N):
            if ((i - j) % 2 == 0):
                BM[i, j] = PascalM[N-1, i] * PascalM[i, j]
            else:
                BM[i, j] = -1.0 * PascalM[N-1, i] * PascalM[i, j]
    return BM

def genConstrM(deg, der):
    Nctrp = deg + 1
    O = np.eye(Nctrp - der, Nctrp)

    if (der == 1):
        tempO = np.eye(Nctrp - der, Nctrp, k = 1)
        O = (O - tempO) * deg

    if (der == 2):
        tempO = np.eye(Nctrp - der, Nctrp, k = 1)
        O = O - 2.0 * tempO

        tempO = np.eye(Nctrp - der, Nctrp, k = 2)
        O = (O + tempO) * deg * (deg - 1)

    return O



def buildInterpolationProblem(X, deg):
    """
    #Build the interpolation problem A * x = b

    Args
        X:   Matrix of the constraints Nconstr x Nwaypoints
        deg; Degree of the polynomial to fit to describe the pieces

    Output:
        A: Matrix of the system
        b: Vector of the known terms
    """

    nCoef = deg + 1        # Number of coefficient to describe each polynomial
    nConstr = X.shape[0]   # Number of constraints (flat outputs)
    nWp = X.shape[1]       # Number of waypoints
    nEq = nConstr * 2      # Number of equations 

    print("Degree of the Bezier curve = " + str(deg))
    print("Number of control points = " + str(nCoef))
    print("Number of equality constraints = " + str(nEq))


    # Instantiate the output variables
    A = np.zeros((nEq, nCoef), dtype=float)
    b = np.zeros((nEq), dtype=float)

    # Define the time vector
    Dt = np.ones((nWp - 1), dtype=float)

    # The bezier polynomial can be represented using the 
    # standard polynomial power basis [1 t t^2 ...].
    # In this form it is written as T * M, where T is the 
    # power basis vector and M is a matrix with binomial 
    # coefficients.
    
    M = genBezierM(deg)

    counter = 0;
    for i in range(nWp):
        for j in range(nConstr):
            if (i == 0):
                b[counter] = X[j, i]
                v = trj_core.polyder(trj_core.t_vec(0, deg), j)
                A[counter, :] = np.matmul(v, M)
            else:
                b[counter] = X[j,i]
                v = trj_core.polyder(trj_core.t_vec(Dt[i-1], deg), j)
                A[counter, :] = np.matmul(v, M)
            counter = counter + 1
    return (A,b)



def costFun(x):
    """
    Generate the const function for the optimization problem
    """
    deg = x.size - 1
    M = genBezierM(deg)
    Q = trj_core.genQ([1.0], deg, 4)

    Q = M.transpose() * Q * M

    cost = np.matmul(x, Q).dot(x)

    return cost

def bz_jac(x):
    deg = x.size - 1
    M = genBezierM(deg)
    Q = trj_core.genQ([1.0], deg, 4)

    Q = M.transpose() * Q * M

    return 2.0 * np.matmul(x, Q)

def bz_hess(x):
    deg = x.size - 1
    M = genBezierM(deg)
    Q = trj_core.genQ([1.0], deg, 4)
    return 2.0 * M.transpose() * Q * M


def genBezier(wp, constr, deg):
    """
    wp: Matrix Nder x Npoints
    constr: Nder x [lb ub]
    """
    nCoeff = deg + 1
    nConstr = constr.shape[0]

    (A, b) = buildInterpolationProblem(wp, deg)

    # Build the constraint list
    lin_constr = []

    # Interpolation Constraints
    lin_constr = np.append(lin_constr, LinearConstraint(A, b, b))

    # Limit Constraints
    for i in range(nConstr):
            CM = genConstrM(deg, i)
            clow = np.ones(CM.shape[0]) * constr[i, 0]
            cup  = np.ones(CM.shape[0]) * constr[i, 1]

            lin_constr = np.append(lin_constr, LinearConstraint(CM, clow, cup))

    x0 = np.zeros((nCoeff), dtype=float)
    res = minimize(costFun, x0, method='trust-constr', constraints=lin_constr, options={'verbose':1})

    return res



class Bezier :

    ## Constructor
    def __init__(self, cntp=None, waypoints=None, constraints=None, degree=None):
        # Asking for interpolation
        if (waypoints is not None and degree is not None):
            # Store the waypoints
            self.wp = waypoints

            # Store the constraints
            self.cnstr = constraints

            # Degree of the polynomial
            self.degree = degree

            # Interpolation problem
            sol = genBezier(self.wp, self.cnstr, self.degree)

            print("Control Points:")
            print(sol.x)

            # Control points of the bezier curve
            self.cntp = np.array(sol.x)

        elif (cntp is not None):
            self.cntp = cntp

            self.degree = cntp.size - 1

    # Evaluate the Bezier polynomial
    def eval(self, t, der = [0]):
        """
        Evaluate the Bezier polynomial

        Args:
            t time
            der derivative to evaluate
        """

        if (t < 0.0) or (t > 1.0):
            print("Warning!  to evaluate outside " +
                    "the time support of the Bezier polynomial")
            return np.nan

        # Create the time vector and the derivatives
        N_der = len(der)

        t_ = trj_core.t_vec(t, self.degree)

        M = genBezierM(self.degree)

        t_der = np.zeros((N_der, t_.size))
        yout = np.zeros(N_der)
        for d in range(N_der):
            t_der[d, :] = trj_core.polyder(t_, der[d])
            yout[d] = np.matmul(np.matmul(t_der[d,:], M), self.cntp)

        return yout


    ## Function to retrieve values

    def getWaypoints(self):
        """
        Returns the waypoints of the piecewise polynomial
        """
        wp = self.wp
        return np.array(wp)

    def getControlPts(self):
        """
        Returns the control points of the Bezier polynomial
        """
        coeff = self.coeff_m
        return np.array(coeff)

    def getCoeffVect(self):
        """
        Returns the coefficients of the piecewise polynomial
        as a vector
        """
        coeff_v = self.coeff_v
        return coeff_v

    def loadFromData(self, M, Dt, npieces):
        """
        Load the polynomial from data
        """
        pass







