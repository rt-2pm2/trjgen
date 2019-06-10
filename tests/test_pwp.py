#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 02:54:42 2019

Test the functionality of the Piecewise polynomial class in "pwpoly.py"

@author: rt-2pm2
"""
import sys
sys.path.insert(0, '../trjgen')

import numpy as np

from trjgen import trjgen_helpers as tjh
from trjgen import trjgen as tj
from trjgen import pwpoly as pw

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
        [ 0,     1.0,    0.6],
        [ 0,     0.02,   0.0],
        [ 0,   -9.0,     0.0],
        [ 0,   np.nan,   0.0],
        ])

# Times (Absolute and intervals)
knots = np.array([0, 1.5, 3]) # One second each piece
Dt = knots[1:len(knots)] - knots[0:len(knots)-1]

# Generate the polynomial
ppx = pw.PwPoly(X, knots, ndeg)

wps = ppx.getWaypoints()
print("Waypoints: ", wps)
knts = ppx.getKnots()
print("Knots: ", knts)
cfs = ppx.getCoeff()
print("Coefficients: ", cfs)


print('\n')
knots = np.array([0, 2, 3.5])
print("Changing the knots position: ", knots)
ppx.moveKnots(knots)
cfs = ppx.getCoeff()
print("Coefficients: ", cfs)

X = np.array([
        [ 0,     2.0,    3.6],
        [ 0,     0.05,   0.0],
        [ 0,   -9.5,     0.0],
        [ 0,   np.nan,   0.0],
        ])

ppx.moveWps(X)
wps = ppx.getWaypoints()
print("Waypoints: ", wps)
knts = ppx.getKnots()
print("Knots: ", knts)
cfs = ppx.getCoeff()
print("Coefficients: ", cfs)


