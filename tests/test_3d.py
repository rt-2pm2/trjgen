#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test the trajectory generation library
Example with 3D trajectory

@author: Luigi Pannocchi
"""

import numpy as np
from  trjgen import trjgen as tj
#from .context import trjgen

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
t = np.array([0, 3, 5, 6, 7])
Dt = t[1:len(t)] - t[0:len(t)-1]

toff = np.zeros(len(Dt) + 1);
toff[1: len(Dt) + 1] = np.matmul(np.tril(np.ones((len(Dt), len(Dt))), 0), np.array(Dt))

# Build the constraint matrix
X = np.array([
        [ 0,   0.5,      1.0,     1.3,    1.3],
        [ 0,   0.0,     np.nan,   0.1,    0.0],
        [ 0,   np.nan,  np.nan,  -2.0,    0.0],
        [ 0,   np.nan,  np.nan, np.nan,   0.0],
        ])

Y = np.array([
        [ 0,   0.0,     0.0  ,    0.0,    0.0],
        [ 0,   0.0,    np.nan,  np.nan,   0.0],
        [ 0,   0.0,    np.nan,  np.nan,   0.0],
        [ 0,   np.nan, np.nan,  np.nan,   0.0],
        ])

Z = np.array([
        [ 0,   0.0,     0.2,      0.2,    0.0],
        [ 0,   0.0,    np.nan,  np.nan,   0.0],
        [ 0,   0.0,    np.nan,   1.0,     0.0],
        [ 0,   np.nan,  np.nan, np.nan,   0.0],
        ])



(solx, nullx, resx, polysx) = tj.interpolPolys(X, ndeg, Dt, abstime=False)
(soly, nully, resy, polysy) = tj.interpolPolys(Y, ndeg, Dt, abstime=False)
(solz, nullz, resz, polysz) = tj.interpolPolys(Z, ndeg, Dt, abstime=False)



if (resx.size>0 and abs(resx) > 0.001):
    print("WARNING! PROBABLY YOU WON\'T BE SATISFIED WITH THE SOLUTION")
if (resy.size>0 and abs(resy) > 0.001):
    print("WARNING! PROBABLY YOU WON\'T BE SATISFIED WITH THE SOLUTION")
if (resz.size>0 and abs(resz) > 0.001):
    print("WARNING! PROBABLY YOU WON\'T BE SATISFIED WITH THE SOLUTION")


print("Polynomial for X")
for i in range(polysx.shape[0]):
    print("Polynomial {:d}: \n".format(i), polysx[i, :])

#print("Null Space basis: \n", nullx)

print("Residuals: \n", resx)

print("Polynomial for Y")
for i in range(polysy.shape[0]):
    print("Polynomial {:d}: \n".format(i), polysy[i, :])

#print("Null Space basis: \n", nully)

print("Residuals: \n", resy)

print("Polynomial for Z")
for i in range(polysz.shape[0]):
    print("Polynomial {:d}: \n".format(i), polysz[i, :])

#print("Null Space basis: \n", nullz)

print("Residuals: \n", resz)
