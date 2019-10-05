import sys
import numpy as np
import numpy.polynomial.polynomial as poly


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trjgen.class_bz as bz
from trjgen.class_bz import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


def computeDelta(coeff):
        c = coeff[0]
        b = coeff[1]
        a = coeff[2]

        delta = b**2 - 4.0 * a * c 

        return delta

if __name__ == "__main__":
    
    NCPPoints = 6
    NCVPoints = NCPPoints - 1
    NCAPoints = NCVPoints - 1
    P = np.zeros(NCPPoints)
    V = np.zeros(NCVPoints)
    A = np.zeros(NCAPoints)

    print("Passing ")

    Pf = float(sys.argv[1])
    print("Final Point: ", Pf)
    Vf = float(sys.argv[2])
    print("Final Velocity: ", Vf)
    Af = float(sys.argv[3])
    print("Final Acceleration: ", Af)
    t = float(sys.argv[4])
    print("Final Time: ", t)

    if (len(sys.argv) > 5):
        Al = float(sys.argv[5])
        print("Lower bound: ", Al)
        Au = float(sys.argv[6])
        print("Upper bound: ", Au)
    
    P[0] = 0.0
    P[1] = 0.0
    P[2] = 0.0
    P[3] = Pf - (2.0 / 5.0) * Vf + t**2 * Af / 20.0
    P[4] = Pf - t / 5.0 * Vf 
    P[NCPPoints - 1] = Pf

    A[0] = 0.0
    A[1] = Af - (8.0 * Vf / t) + (20.0 / t**2) * Pf
    A[2] = - 2.0 * Af + (12.0 * Vf / t) - (20.0 / t**2) * Pf
    A[NCAPoints - 1] = Af
    
    V[0] = 0.0
    V[1] = 0.0
    V[2] = (Af * t) / 4.0 - 2.0 * Vf + (5.0 / t) * Pf
    V[3] = -(Af * t) / 4.0 + Vf   
    V[NCVPoints - 1] = Vf

    
    # Condition for A[1] < Ul
    coeff_ub1 = np.array([20.0 * Pf, -8.0 * Vf, Af - Au])
    roots_ub1 = poly.polyroots(coeff_ub1)
    delta_ub1 = computeDelta(coeff_ub1)

    # Condition for A[1] > Ll
    coeff_lb1 = np.array([20.0 * Pf, -8.0 * Vf, Af - Al])
    roots_lb1 = poly.polyroots(coeff_lb1)
    delta_lb1 = computeDelta(coeff_lb1)

    # Condition for A[2] < Ul   
    coeff_ub2 = np.array([-20.0 * Pf, 12.0 * Vf, -2.0 * Af - Au])
    roots_ub2 = poly.polyroots(coeff_ub2)
    delta_ub2 = computeDelta(coeff_ub2)

    # Condition for A[2] > Ll
    coeff_lb2 = np.array([-20.0 * Pf, 12.0 * Vf, -2.0 * Af - Al])
    roots_lb2 = poly.polyroots(coeff_lb2)
    delta_lb2 = computeDelta(coeff_lb2)


    print("\nPosition Control Points")
    print(P)
    print("\nVelocity Control Points")
    print(V)
    print("\nAcceleration Control Points")
    print(A)

       
    print("\nTime Analysis")
    
    print("A1 UB: a coeff", coeff_ub1)
    if (coeff_ub1[2] > 0.0):
        if (delta_ub1 < 0):
            print("Never Satisfied")
        else:
            print("{} < t < {}".format(roots_ub1[0], roots_ub1[1]))
    else:
        if (delta_ub1 < 0):
            print("Always Satisfied")
        else:
            print("t < {}, t > {}".format(roots_ub1[0], roots_ub1[1]))
    
    print("A1 LB: a coeff", coeff_lb1)
    if (coeff_lb1[2] > 0.0):
        if (delta_lb1 < 0):
            print("Always Satisfied")
        else:
            print("t < {}, t > {}".format(roots_lb1[0], roots_lb1[1]))
    else:
        if (delta_lb1 < 0):
            print("Never Satisfied")
        else:
            print("{} < t < {}".format(roots_lb1[0], roots_lb1[1]))


    print("\n") 
    print("A2 UB: a coeff", coeff_ub2)
    if (coeff_ub2[2] > 0.0):
        if (delta_ub2 < 0):
            print("Never Satisfied")
        else:
            print("{} < t < {}".format(roots_ub2[0], roots_ub2[1]))
    else:
        if (delta_ub2 < 0):
            print("Always Satisfied")
        else:
            print("t < {}, t > {}".format(roots_ub2[0], roots_ub2[1]))
 
    print("A2 LB: a coeff", coeff_lb2)
    if (coeff_lb2[2] > 0.0):
        if (delta_lb2 < 0):
            print("Always Satisfied")
        else:
            print("t < {}, t > {}".format(roots_lb2[0], roots_lb2[1]))
    else:
        if (delta_lb2 < 0):
            print("Never Satisfied")
        else:
            print("{} < t < {}".format(roots_lb2[0], roots_lb2[1]))

