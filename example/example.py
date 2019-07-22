#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:04:56 2019

@author: rt-2pm2
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from matplotlib import pyplot as plt

import trjgen.trjgen as tj
import trjgen.trjgen_helpers as tj_h

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

# Polynomial degree
ndeg = 7;

t = np.array([0, 1, 5])
Dt = t[1:len(t)] - t[0:len(t)-1]

toff = np.zeros(len(Dt) + 1);
toff[1: len(Dt) + 1] = np.matmul(np.tril(np.ones((len(Dt), len(Dt))), 0), np.array(Dt))

# Build the constraint matrix
X = np.array([
        [ 0,   1.0,      2.0],
        [ 0,   0.01,      0.0],
        [ 0, -10.0,      0.0],
        [ 0,   np.nan,   0.0],
        ])

Y = np.array([
        [ 0,   0.0,      0.0],
        [ 0,   0.0,      0.0],
        [ 0,   0.0,      0.0],
        [ 0,   np.nan,   0.0],
        ])

Z = np.array([
        [ 0,   0.0,     -0.2],
        [ 0,   0.0,      0.0],
        [ 0,   0.0,      0.0],
        [ 0,   np.nan,   0.0],
        ])


# Compute the polynomials
(solx, nullx, resx, polysx) = tj.interpolPolys(X, ndeg, Dt, False)
(soly, nully, resy, polysy) = tj.interpolPolys(Y, ndeg, Dt, False)
(solz, nullz, resz, polysz) = tj.interpolPolys(Z, ndeg, Dt, False)


# Compute the trajectory from the polynomials
(T, Xtj, Ytj, Ztj) = tj_h.polysTo3D(Dt, 100, polysx, polysy, polysz, [0,1,2])


#tj.plotPoly3D(Dt, 1000, polysx, polysy, polysz, [0])

plt.figure();
axx = plt.subplot(3, 1, 1)
axx.plot(T, Xtj[0, :])
axx.set_title("X")
axx.set_xlabel("time [s]")
axx.grid(True)

axy = plt.subplot(3, 1, 2)
axy.plot(T, Ytj[0, :])
axy.set_title("Y")
axy.set_xlabel("time [s]")
axy.grid(True)

axz = plt.subplot(3, 1, 3)
axz.plot(T, Ztj[0, :])
axz.set_title("Z")
axz.set_xlabel("time [s]")
plt.tight_layout()
axz.grid(True)

# Velocity
plt.figure();
axx = plt.subplot(3, 1, 1)
axx.plot(T, Xtj[1, :])
axx.set_title("Xd")
axx.set_xlabel("time [s]")
axx.grid(True)

axy = plt.subplot(3, 1, 2)
axy.plot(T, Ytj[1, :])
axy.set_title("Yd")
axy.set_xlabel("time [s]")
axy.grid(True)

axz = plt.subplot(3, 1, 3)
axz.plot(T, Ztj[1, :])
axz.set_title("Zd")
axz.set_xlabel("time [s]")
plt.tight_layout()
axz.grid(True)

# Acceleration
plt.figure();
axx = plt.subplot(3, 1, 1)
axx.plot(T, Xtj[2, :])
axx.set_title("Xdd")
axx.set_xlabel("time [s]")
axx.grid(True)

axy = plt.subplot(3, 1, 2)
axy.plot(T, Ytj[2, :])
axy.set_title("Ydd")
axy.set_xlabel("time [s]")
axy.grid(True)

axz = plt.subplot(3, 1, 3)
axz.plot(T, Ztj[2, :])
axz.set_title("Zdd")
axz.set_xlabel("time [s]")
plt.tight_layout()
axz.grid(True)

plt.show()
#(ffthrust, available_thrust) = tj_h.plotThrustMargin(T, Xtj, Ytj, Ztj, 0.0032, 9.91 * 0.032 * 2)
