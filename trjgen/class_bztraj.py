#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rt-2pm2
"""
import numpy as np

class BezierCurve:
    """
    Class representing Bezier Curves
    """

    ## Constructor
    def __init__(self, px, py, pz, pw):
        self.px = px
        self.py = py
        self.pz = pz
        self.pw = pw


    ## Evaluate the trajectory at a time 't' over a list
    ## of derivative 'derlist'
    def eval(self, t, derlist):
        # Check whether the input is a scalar
        if type(derlist) == int:
            der = np.array([derlist])
        else:
            der = np.array(derlist)

        X = np.zeros((der.size))
        Y = np.zeros((der.size))
        Z = np.zeros((der.size))
        W = np.zeros((der.size))

        # Evaluate the polynomials on the requested
        # derivatives
        for i in range(der.size):
            X[i] = self.px.eval(t, [der[i]])
            Y[i] = self.py.eval(t, [der[i]])
            Z[i] = self.pz.eval(t, [der[i]])
            W[i] = self.pw.eval(t, [der[i]])

        # If the acceleration has been required, it's possible to compute
        # the demanded orientation of the vehicle

        R = np.zeros((3,3), dtype=float)
        Xb = np.zeros((3), dtype=float)
        Yb = np.zeros((3), dtype=float)
        Zb = np.zeros((3), dtype=float)
        Omega = np.zeros((3), dtype=float)

        if (np.array(derlist) >= 2).any():

            # Assemble the acceleration 3d Vector
            Thrust = np.array((X[2], Y[2], Z[2] + 9.81))

            # Find the demanded Zb axis
            Zb = (Thrust / np.linalg.norm(Thrust))

            X_w = np.array((np.cos(W[0]), np.sin(W[0])))
            Yb = np.cross(Zb, X_w)
            Yb = Yb / np.linalg.norm(Yb)
            Xb = np.cross(Yb, Zb)

            # Compute the rotation matrix associated with the body frame
            R[:, 0] = Xb
            R[:, 1] = Yb
            R[:, 2] = Zb

        if (np.array(derlist) >= 3).any():
            u1 = np.linalg.norm(Thrust)
            a_dot = np.array([X[3], Y[3], Z[3]], dtype=float)

            if (u1 > 0.01):
                h_omega = (a_dot - np.dot(Zb, a_dot) * Zb) / u1
            else:
                h_omega = np.zeros((3), dtype=float)

            # Omega_x
            Omega[0] = -np.dot(h_omega, Yb)
            Omega[1] = np.dot(h_omega, Xb)
            Omega[2] = W[1] * Zb[2]


        return (X, Y, Z, W, R, Omega)
