"""
Created on Sat Jun  8 02:54:42 2019

Test the functionality of the Bezier polynomial class in "class_bz.py"

@author: rt-2pm2
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trjgen.class_bz as bz

from trjgen.class_bz import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)



##### Testing Helper Functions
print("\nMapping Matrix")
for i in range(1,5):
    print("M({}) = ".format(i))
    print(genBezierM(i))

ndeg = 6
print("\nConstr Matrices for {} deg poly".format(ndeg))
for i in range(ndeg):
    print("Constr matrix for {} derivative".format(i))
    print(genConstrM(ndeg, i))

print("\nDerivative Matrix for {} deg".format(ndeg))
for i in range(ndeg):
    print("Order = {}".format(i))
    print(genDM(ndeg, i))

print("\nCost Matrix for {} deg".format(ndeg))
for i in range(ndeg - 1):
    print("Order = {}".format(i))
    print(genCostM(ndeg, i))


print("\n \n \n")

# Build the waypoint matrix
X = np.array([
        [ 0,  0.5], # p
        [ 0,  0.0], # v
        [ 0,  -3.5], # a
        ])

x_cnstr = np.array([[-0.5, 1.0], [-5.0, 5.0], [-25.0, 25.0], [-700, 700]])

# Generate the polynomial
T = 1.0
bz_x = bz.Bezier(waypoints=X, constraints=x_cnstr, degree=8, s=T, opt_der = 4)

print("Evaluation of the bezier polynomial")
print(bz_x.eval(1.0, [0,1,2]))



#### PLOT
N = 100
test_y = np.zeros((N, 3), dtype=float)
t = np.zeros((N), dtype=float)

Xtj = np.zeros((N, 4), dtype=float)
for i in range(N):
    t[i] = T/(N - 1) * i
    Xtj[i, :] = bz_x.eval(t[i], [0,1,2,3])

fig, axs = plt.subplots(4, 1)
axs[0].plot(t, Xtj[:, 0], t, np.ones(N) * x_cnstr[0,0], t, np.ones(N) * x_cnstr[0,1], T, X[0,1], 'o')
axs[0].set_title("p")

axs[1].plot(t, Xtj[:, 1], t, np.ones(N) * x_cnstr[1,0], t, np.ones(N) * x_cnstr[1,1], T, X[1,1], 'o')
axs[1].set_title("v")

axs[2].plot(t, Xtj[:, 2], t, np.ones(N) * x_cnstr[2,0], t, np.ones(N) * x_cnstr[2,1], T, X[2,1], 'o')
axs[2].set_title("a")

axs[3].plot(t, Xtj[:, 3], t, np.ones(N) * x_cnstr[3,0], t, np.ones(N) * x_cnstr[3,1])
axs[3].set_title("a")


plt.show()
