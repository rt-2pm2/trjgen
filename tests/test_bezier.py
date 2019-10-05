"""
Created on Sat Jun  8 02:54:42 2019

Test the functionality of the Bezier polynomial class in "class_bz.py"

@author: rt-2pm2
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import scipy.linalg as linalg

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

ndeg = 8 
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
        [ 0,  0.0], # a
        ])

x_cnstr = np.array([[-0.5, 1.0], [-5.0, 5.0], [-25.0, 25.0]])



T = 1.0
## Testing the components of the Bezier polynomial generation:
(A, b) = buildInterpolationProblem(X, ndeg, T)

print("A = ")
print(A)

print("b = ")
print(b)

def null_space(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

nullx = null_space(A)
if (nullx.size == 0):
    print("Null space is void")

lstq_sol = np.matmul(np.linalg.pinv(A),b)

print("Lstqr Solution:")
print(lstq_sol)

print("Test A * sol = b")
print(np.matmul(A, lstq_sol))


print("Null Space = ")
print(nullx)

# Generate the polynomial
D = genDM(ndeg, 3)
Q = np.matmul(np.transpose(D), D)
Q = Q / Q.max()
P = 2.0 * np.matmul(np.matmul(lstq_sol, Q), nullx) 
Q = np.matmul(np.matmul(np.transpose(nullx), Q), nullx) 

G = np.empty(shape=(0, nullx.shape[1]))
h = []
nConstr = x_cnstr.shape[0]
for i in range(nConstr):
    CM = genConstrM(ndeg, i, T)
    G_ = np.matmul(CM, nullx)
    G =  np.vstack((G, -G_, G_)) 

    temp = np.matmul(CM, lstq_sol)
    clow = np.ones(CM.shape[0]) * x_cnstr[i, 0] - temp
    cup  = np.ones(CM.shape[0]) * x_cnstr[i, 1] + temp
    h = np.concatenate((h, -clow, cup))

print("D = ")
print(D)
print("Q = ")
print(Q)
print("P = ")
print(P)
print("G = ")
print(G)
print("h = ")
print(h)

x_size = nullx.shape[1]
Qcvx = matrix(Q)
Pcvx = matrix(P)
Gcvx = matrix(G)
hcvx = matrix(h)

my_sol_ = solvers.qp(P = Qcvx, q = Pcvx , G = Gcvx, h = hcvx)
my_sol = my_sol_['x']
my_sol_full = lstq_sol + np.matmul(nullx, my_sol).flatten()
print("Manual solution, removing equality constratins:")
print(my_sol_full)

print("\nChecking....")
print("A x = b")
print(np.matmul(A, my_sol_full) - b)

print("G x < h")
print(np.matmul(G, my_sol) - h)

print("\n\n")
print("Class Construction")
bz_x = bz.Bezier(waypoints=X, constraints=x_cnstr, degree=ndeg, s=T, opt_der = 3)

#
#print("Evaluation of the bezier polynomial")
#print(bz_x.eval(1.0, [0,1,2]))
#
#
#
##### PLOT
#N = 100
#test_y = np.zeros((N, 3), dtype=float)
#t = np.zeros((N), dtype=float)
#
#Xtj = np.zeros((N, 4), dtype=float)
#for i in range(N):
#    t[i] = T/(N - 1) * i
#    Xtj[i, :] = bz_x.eval(t[i], [0,1,2,3])
#
#fig, axs = plt.subplots(4, 1)
#axs[0].plot(t, Xtj[:, 0], t, np.ones(N) * x_cnstr[0,0], t, np.ones(N) * x_cnstr[0,1], T, X[0,1], 'o')
#axs[0].set_title("p")
#
#axs[1].plot(t, Xtj[:, 1], t, np.ones(N) * x_cnstr[1,0], t, np.ones(N) * x_cnstr[1,1], T, X[1,1], 'o')
#axs[1].set_title("v")
#
#axs[2].plot(t, Xtj[:, 2], t, np.ones(N) * x_cnstr[2,0], t, np.ones(N) * x_cnstr[2,1], T, X[2,1], 'o')
#axs[2].set_title("a")
#
#axs[3].plot(t, Xtj[:, 3], t, np.ones(N) * x_cnstr[3,0], t, np.ones(N) * x_cnstr[3,1])
#axs[3].set_title("a")
#
#
#plt.show()
