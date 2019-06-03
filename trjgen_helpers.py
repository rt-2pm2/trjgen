#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions


@author: rt-2pm2
"""
import numpy as np
import numpy.polynomial.polynomial as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def jointPieces(Dt, Nsamples, polys, der_list):
    # Number of derivates to evaluate
    numDer = len(der_list)

    # Number of pieces to join
    numPieces = len(Dt)

    # Time offsets
    toff = np.zeros(numPieces + 1);
    toff[1: numPieces + 1] = np.matmul(np.tril(np.ones((numPieces, numPieces)), 0), np.array(Dt))

    # Matrix of time and output vector to represent the pieces
    t = np.zeros((numPieces, Nsamples))
    y = np.zeros((numPieces, Nsamples))

    # The overall time vector to include all the pieces
    #T = np.linspace(0.0, toff[-1], numPieces * Nsamples)
    T = np.zeros((numPieces * Nsamples))
    Y = np.zeros((numDer, numPieces * Nsamples))


    # Evaluation
    for i in range(len(Dt)):
        t[i, :] = np.linspace(0.0, Dt[i], Nsamples)
        for j in range(numDer):
            # Evaluate the polynomial
            pder = pl.polyder(polys[i,:], der_list[j])
            y[i,:] = pl.polyval(t[i,:], pder)
            # Compose the single vectors
            Y[j, i*Nsamples: (i+1)*Nsamples] = y[i, :]
            T[i*Nsamples: (i+1)*Nsamples] = t[i,:]

    return (T, Y)



def polysTo3D(Dt, Nsamples, polysx, polysy, polysz, der):

    [T, X] = jointPieces(Dt, Nsamples, polysx, der)
    [_, Y] = jointPieces(Dt, Nsamples, polysy, der)
    [_, Z] = jointPieces(Dt, Nsamples, polysz, der)

    return (T, X, Y, Z)


def plotPoly3D(Dt, Nsamples, polx, poly, polz, derlist):

    (T, X, Y, Z) = polysTo3D(Dt, Nsamples, polysx=polx, polysy=poly, polysz=polz, der=derlist)

    for i in range(len(derlist)):
        # Create the figure and axes
        plt.figure(i)
        ax = plt.axes(projection='3d')
        title_str = 'Derivative n = {:1d}'.format(derlist[i])
        print(title_str)
        ax.set_title(title_str)
        if derlist[i] == 0:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")
        if derlist[i] == 1:
            ax.set_xlabel("x [m/s]")
            ax.set_ylabel("y [m/s]")
            ax.set_zlabel("z [m/s]")
        if derlist[i] == 2:
            ax.set_xlabel("x [m/s^2]")
            ax.set_ylabel("y [m/s^2]")
            ax.set_zlabel("z [m/s^2]")

        ax.scatter(X[i, :], Y[i, :], Z[i, :], c=T)
