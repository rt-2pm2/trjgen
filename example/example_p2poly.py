#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 02:54:42 2019

@author: rt-2pm2
"""
import sys
sys.path.insert(0, '../trjgen')

import numpy as np

import plotly as py
import plotly.graph_objs as go

from trjgen import trjgen_helpers as tjh
from trjgen import pltly_helpers as ply
from trjgen import trjgen as tj
from trjgen import pwpoly as pw

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


# Vehicle Data
vehicle_mass = 0.032
thust_thr = 9.91 * 0.032 * 2.0

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

Y = np.array([
        [ 0,   0.0,      0.0],
        [ 0,   0.0,      0.0],
        [ 0,   0.0,      0.0],
        [ 0,   np.nan,   0.0],
        ])

Z = np.array([
        [ 0,   0.0,     -0.3],
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

# Times (Absolute and intervals)
knots = np.array([0, 1.5, 3]) # One second each piece

# Generate the polynomial
ppx = pw.PwPoly(X, knots, ndeg)
ppy = pw.PwPoly(Y, knots, ndeg)
ppz = pw.PwPoly(Z, knots, ndeg)
ppw = pw.PwPoly(W, knots, ndeg)

# Check (Evaluate polynomial)
tv = np.linspace(0,max(knots),100);
(Xtj, Ytj, Ztj, Wtj, Zbtj) = tjh.TrajFromPW(tv, [0,1,2], \
        pwpolx=ppx, pwpoly=ppy, pwpolz=ppz, pwpolw = ppw)

# Save the polynomial coefficients on file
x_coeff = ppx.getCoeffMat(); 
y_coeff = ppy.getCoeffMat(); 
z_coeff = ppz.getCoeffMat(); 
w_coeff = ppy.getCoeffMat();

Dt = knots[1:len(knots)] - knots[0:len(knots)-1]
tj.pp2file(Dt, x_coeff, y_coeff, z_coeff, w_coeff, "./poly.csv")

ply.plotTray_plotly(Xtj, Ytj, Ztj, tv)

ply.plotThrustMargin(tv, Xtj, Ytj, Ztj, vehicle_mass, thust_thr)
