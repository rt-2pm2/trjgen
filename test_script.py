#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test the trajectory generation library

@author: Luigi Pannocchi
"""

import numpy as np
import trjgen as tj

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


# Polynomial characteristic: 4th order
ndeg = 4
print('Test with {:d}th order polynomial.'.format(ndeg))

## Waipoints in the flat output space (or dimension 3)
nconstr = 3
print('Number of constraint on the flat output = {:d}.'.format(nconstr))
s = np.zeros((nconstr, 1), dtype=float)
sf = np.zeros((nconstr, 1), dtype=float)

# Initial Pointt
t0 = 0
s0 = np.zeros((nconstr, 1), dtype=float)
s0[0] = 0.0

# Intermediate point
t1 = 8
si1 = np.empty((nconstr,1))
si1[:] = np.nan
si1[2] = -0.2

# Final point
t2 = 15
sf[0] = 1.0
sf[2] = 0.0


t = np.array([t0, t1, t2]);
Dt = np.array([t1-t0, t2-t1]);

# Build the constraint matrix
X = np.concatenate((s0, si1, sf), axis=1)
print("Constraints matrix = \n", X)

print('Waypoints')
print('{:1s} | {:^5s}| {:^5s}| {:^5s}'.format("N","x","x\'","x\'\'"))
for i in range(X.shape[1]):
    print('{:1d} | {:^5.2f}, {:^5.2f}, {:^5.2f}'.format(i,
            X[0][i],
            X[1][i],
            X[2][i]))


[A, b] = tj.buildInterpolationProblem(X, ndeg, Dt, False)
print("Interpolation Problem Ax = b: ")
print("A = \n", A)
print("b = \n", b)
print('\n')


print("Solution to the problem: ")
(sol, nullx, res, polys) = tj.interpolPolys(X, ndeg, Dt, abstime=False)

print("Polynomial 1: \n", polys[0, :])
print("Polynomial 2: \n", polys[1, :])

print("Null Space basis: \n", nullx)

print("\n-------- Test -------")
S_test = np.zeros((nconstr, 4))

T0 = tj.constrMat(tj.t_vec(t0, ndeg), [0,1,2])
S_test[:,0] = np.matmul(T0, polys[0,:])

T1 = tj.constrMat(tj.t_vec(t1, ndeg), [0,1,2])
S_test[:,1] = np.matmul(T1, polys[0,:])

T1 = tj.constrMat(tj.t_vec(0, ndeg), [0,1,2])
S_test[:,2] = np.matmul(T1, polys[1,:])

T2 = tj.constrMat(tj.t_vec(t2-t1, ndeg), [0,1,2])
S_test[:,3] = np.matmul(T2, polys[1,:])

print('{:1s} | {:^5s}| {:^5s}| {:^5s}'.format("N","x","x\'","x\'\'"))
for i in range(4):
    print('{:1d} | {:^5.2f}, {:^5.2f}, {:^5.2f}'.format(i,
            S_test[0,i],
            S_test[1,i],
            S_test[2,i]))