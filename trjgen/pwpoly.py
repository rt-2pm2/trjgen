#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:11:29 2019

@author: rt-2pm2
"""
import numpy as np
import numpy.polynomial.polynomial as pl



class PwPoly :

    ## Constructor
    def __init__(self, coeff_m, knots):

        # Matrix of polynomials coefficients
        self.coeff = np.array(coeff_m)

        # Number of pieces
        self.npieces = coeff_m.shape[0]

        # Knots (It's assumed that the knot of the first point is 0.0)
        self.knots = np.array(knots)

        # Degree of the polynomial
        self.deg = coeff_m.shape[1] - 1


    # Evaluate the Piecewise polynomial
    def eval(self, t, der):
        """
        Evaluate the piecewise polynomial

        Args:
            t vector of time instants
            der derivative to evaluate
        """

        # Check whether the evaluation is requested on a scalar or on an
        # iterable
        try:
            # Iterable: t is a vector
            t = np.array(t)
            _ = iter(t)
            nsamples = t.size
            if (t[0] < 0.0) or (t[nsamples - 1] > self.knots[-1]):
                print("PwPoly.eval(): Trying to evaluate outside " +
                        "the time support of the piecewise polynomial")

            # Allocate the variable
            yout = np.zeros((nsamples), dtype = float)
            for (k, t_) in enumerate(t):
                i = self.find_piece(t_)
                # Evaluate the required polynomial derivative
                pder = pl.polyder(self.coeff[i, :], der)
                yout[k] = pl.polyval(t_ - self.knots[i], pder)
        except TypeError:
            # Non iterable: t is a single value
            if (t < 0.0) or (t > self.knots[-1]):
                print("Warning! PwPoly.eval(): Trying to evaluate outside " +
                        "the time support of the piecewise polynomial")

            i = self.find_piece(t)
            pder = pl.polyder(self.coeff[i, :], der)
            yout = pl.polyval(t - self.knots[i], pder)

        return yout

    ## Private
    # Find the piece of the piecewies polynomial active at time t
    def find_piece(self, t):
        if (t < self.knots[0] or t > self.knots[-1]):
            return -1

        for i in range(self.npieces):
            if t >= self.knots[i] and t <= self.knots[i+1]:
                return i




