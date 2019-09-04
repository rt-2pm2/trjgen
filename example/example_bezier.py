"""
Created on Sat Jun  8 02:54:42 2019

Test the functionality of the Bezier polynomial class in "class_bz.py"

@author: rt-2pm2
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trjgen.class_bz as bz
import trjgen.class_bztraj as bz_t
import trjgen.pltly_helpers as ply

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

# Build the waypoint matrix
X = np.array([
        [ 0,  1.28], # p
        [ 0,  0.05], # v
        [ 0,  -0.6], # a
        ])

Y = np.array([
        [ 0,  -2.25], # p
        [ 0,  -0.71], # v
        [ 0,  8.03], # a
        ])

Z = np.array([
        [ 0,  0.05], # p
        [ 0,  -0.36], # v
        [ 0,  -5.79], # a
            ])

W = np.array([
        [ 0,  0.0], # p
        [ 0,  0.0], # v
        [ 0,  0.0], # a
        ])


x_lim = 3.0
v_lim = 5.0
a_lim = 10.0

x_cnstr = np.array([[-x_lim, x_lim], [-v_lim, v_lim], [-a_lim, a_lim]])
y_cnstr = np.array([[-x_lim, x_lim], [-v_lim, v_lim], [-a_lim, a_lim]])
z_cnstr = np.array([[-0.5, 1.5], [-v_lim, v_lim], [-a_lim, a_lim]])
w_cnstr = np.array([[0.0, 1.2],  [-v_lim, v_lim], [-a_lim, a_lim]])

# Generate the polynomial
T = 4.0
bz_x = bz.Bezier(waypoints=X, constraints=x_cnstr, degree=10, s=T)
bz_y = bz.Bezier(waypoints=Y, constraints=y_cnstr, degree=10, s=T)
bz_z = bz.Bezier(waypoints=Z, constraints=z_cnstr, degree=10, s=T)
bz_w = bz.Bezier(waypoints=W, constraints=w_cnstr, degree=5, s=T)

Curve = bz_t.BezierCurve(bz_x, bz_y, bz_z, bz_w)


#### PLOT
N = 100
Xtj = np.zeros((N, 3), dtype=float)
Ytj = np.zeros((N, 3), dtype=float)
Ztj = np.zeros((N, 3), dtype=float)
Zbtj = np.zeros((N, 3), dtype=float)


t = np.zeros((N), dtype=float)

for i in range(N):
    t[i] = T/(N - 1) * i
    (Xtj[i, :], Ytj[i, :], Ztj[i, :], _, R, _) = Curve.eval(t[i], [0,1,2])
    Zbtj[i, :] = R[:, 2]/np.linalg.norm(R[:,2])

ply.plotTray_plotly(Xtj.transpose(), Ytj.transpose(), Ztj.transpose(), t)
#ply.plotZb_plotly(Xtj.transpose(), Ytj.transpose(), Ztj.transpose(), Zbtj.transpose())


fig, axs_x = plt.subplots(3, 1)
axs_x[0].plot(t, Xtj[:, 0], t, np.ones(N) * x_cnstr[0,0], t, np.ones(N) * x_cnstr[0,1], T, X[0,1], 'o')
axs_x[0].set_title("p")
axs_x[1].plot(t, Xtj[:, 1], t, np.ones(N) * x_cnstr[1,0], t, np.ones(N) * x_cnstr[1,1], T, X[1,1], 'o')
axs_x[1].set_title("v")
axs_x[2].plot(t, Xtj[:, 2], t, np.ones(N) * x_cnstr[2,0], t, np.ones(N) * x_cnstr[2,1], T, X[2,1], 'o')
axs_x[2].set_title("a")


fig, axs_y = plt.subplots(3, 1)
axs_y[0].plot(t, Ytj[:, 0], t, np.ones(N) * y_cnstr[0,0], t, np.ones(N) * y_cnstr[0,1], T, Y[0,1], 'o')
axs_y[0].set_title("p")                                                                                  
axs_y[1].plot(t, Ytj[:, 1], t, np.ones(N) * y_cnstr[1,0], t, np.ones(N) * y_cnstr[1,1], T, Y[1,1], 'o')
axs_y[1].set_title("v")                                                                                  
axs_y[2].plot(t, Ytj[:, 2], t, np.ones(N) * y_cnstr[2,0], t, np.ones(N) * y_cnstr[2,1], T, Y[2,1], 'o')
axs_y[2].set_title("a")

fig, axs_z = plt.subplots(3, 1)
axs_z[0].plot(t, Ztj[:, 0], t, np.ones(N) * z_cnstr[0,0], t, np.ones(N) * z_cnstr[0,1], T, Z[0,1], 'o')
axs_z[0].set_title("p")                                                                                
axs_z[1].plot(t, Ztj[:, 1], t, np.ones(N) * z_cnstr[1,0], t, np.ones(N) * z_cnstr[1,1], T, Z[1,1], 'o')
axs_z[1].set_title("v")                                                                                
axs_z[2].plot(t, Ztj[:, 2], t, np.ones(N) * z_cnstr[2,0], t, np.ones(N) * z_cnstr[2,1], T, Z[2,1], 'o')
axs_z[2].set_title("a")

plt.show()



