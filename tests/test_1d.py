#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test the trajectory generation library

@author: Luigi Pannocchi
"""
import numpy as np
from matplotlib import pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trjgen.trjgen as tj

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

#============================================================================

# Polynomial characteristic:  order
ndeg = 7
print('Test with {:d}th order polynomial.'.format(ndeg))

## Waipoints in the flat output space (or dimension 3)
nconstr = 4
print('Number of constraint on the flat output = {:d}.'.format(nconstr))


# Times
t = np.array([0, 1, 2.5, 3, 4])
Dt = t[1:len(t)] - t[0:len(t)-1]

toff = np.zeros(len(Dt) + 1);
toff[1: len(Dt) + 1] = np.matmul(np.tril(np.ones((len(Dt), len(Dt))), 0), np.array(Dt))

# Build the constraint matrix
X = np.array([
        [ 0,   0.0,     2.0,    2.5,    2.5],
        [ 0,   np.nan,  np.nan, np.nan,  0.0],
        [ 0,     -0.2,  np.nan, np.nan,  0.0],
        [ 0,   np.nan,  np.nan, np.nan,  0.0],
        ])

print('\n')
nW = X.shape[1]
print("Constraints matrix = \n", X)
print('\n')

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

for i in range(nW -1):
    print("Polynomial {:d}: \n".format(i), polys[i, :])

print("Null Space basis: \n", nullx)

print("Residuals: \n", res)

if (res.size>0 and abs(res) > 0.001):
    print("WARNING! PROBABLY YOU WON\'T BE SATISFIED WITH THE SOLUTION")

print("\n-------- Test Some Constraints-------")
S_test = np.zeros((nconstr, 4))


T0 = tj.constrMat(tj.t_vec(0, ndeg), [i for i in range(nconstr)])
S_test[:,0] = np.matmul(T0, polys[0,:])

T1 = tj.constrMat(tj.t_vec(Dt[0], ndeg), [i for i in range(nconstr)])
S_test[:,1] = np.matmul(T1, polys[0,:])

T1 = tj.constrMat(tj.t_vec(0, ndeg), [i for i in range(nconstr)])
S_test[:,2] = np.matmul(T1, polys[1,:])

T2 = tj.constrMat(tj.t_vec(Dt[1], ndeg), [i for i in range(nconstr)])
S_test[:,3] = np.matmul(T2, polys[1,:])

print('{:1s} | {:^5s}| {:^5s}| {:^5s}'.format("N","x","x\'","x\'\'"))
for i in range(4):
    print('{:1d} | {:^5.2f}, {:^5.2f}, {:^5.2f}'.format(i,
            S_test[0,i],
            S_test[1,i],
            S_test[2,i]))


# =======================================================================
# Plots

fig, axes = plt.subplots(nrows=3, ncols=1)
fig.tight_layout()
plt_count = 0

Nsamples = 1000
t = np.zeros((len(Dt), Nsamples))
y = np.zeros((len(Dt), Nsamples))

# Position
for i in range(len(Dt)):
    t[i, :] = np.linspace(0.0, Dt[i], Nsamples)
    y[i,:] = np.polynomial.polynomial.polyval(t[i, :], polys[i, :])
    axes[plt_count].plot(t[i, :] + toff[i], y[i, :])

axes[plt_count].set_xlabel('time (s)')
axes[plt_count].set_ylabel('(m)')
axes[plt_count].set_title('Position')
axes[plt_count].grid(True)
plt_count += 1

# Velocity
for i in range(len(Dt)):
    t[i, :] = np.linspace(0.0, Dt[i], Nsamples)
    pder = np.polynomial.polynomial.polyder(polys[i,:])
    y[i,:] = np.polynomial.polynomial.polyval(t[i,:], pder)
    axes[plt_count].plot(t[i,:] + toff[i], y[i,:])

axes[plt_count].set_xlabel('time (s)')
axes[plt_count].set_ylabel('(m/s)')
axes[plt_count].set_title('Velocity')
axes[plt_count].grid(True)
plt_count += 1

# Acceleration
for i in range(len(Dt)):
    t[i, :] = np.linspace(0.0, Dt[i], Nsamples)
    pder2 = np.polynomial.polynomial.polyder(polys[i,:], m=2)
    y[i,:] = np.polynomial.polynomial.polyval(t[i,:], pder2)
    axes[plt_count].plot(t[i,:] + toff[i], y[i,:])

axes[plt_count].set_xlabel('time (s)')
axes[plt_count].set_ylabel('(m/s^2)')
axes[plt_count].set_title('Acceleration')
axes[plt_count].grid(True)

