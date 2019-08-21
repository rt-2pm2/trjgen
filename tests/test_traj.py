#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the functionality of the Trajectory class in "trajectory.py"

@author: rt-2pm2
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trjgen.class_pwpoly as pw
import trjgen.class_trajectory as tr

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# Polynomial characteristic:  order
ndeg = 7
print('Test with {:d}th order polynomial.'.format(ndeg))

## Waipoints in the flat output space (or dimension 3)
nconstr = 4
print('Number of constraint on the flat output = {:d}.'.format(nconstr))


# Build the constraint matrix
X = np.array([
        [ 0,     1.0,    0.5],
        [ 0,     0.8,   0.0],
        [ 0,   -9.0,     0.0],
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

W = np.array([
        [ 0,   np.nan,      0.0],
        [ 0,   np.nan,      0.0],
        [ 0,   np.nan,      0.0],
        [ 0,   np.nan,      0.0],
        ])

mid_time = 1.5
end_time = 2.0

# Times (Absolute and intervals)
knots = np.array([0, mid_time, end_time]) # One second each piece
Dt =  knots[1:3] - knots[0:2]
# Generate the polynomial
ppx = pw.PwPoly(X, knots, ndeg)
ppy = pw.PwPoly(Y, knots, ndeg)
ppz = pw.PwPoly(Z, knots, ndeg)
ppw = pw.PwPoly(W, knots, ndeg)

traj = tr.Trajectory(ppx, ppy, ppz, ppw)

print("Evaluating Trajectory in 0")
print(traj.eval(0.0, [0]))

print("\n")
print("Evaluating trajectory at the end point:")
print(traj.eval(end_time, [0,1,2]))

traj.writeTofile(Dt, filename='./trjfile.csv')
traj.readFromfile(filename='./trjfile.csv')
print(traj.eval(end_time, [0,1,2]))


