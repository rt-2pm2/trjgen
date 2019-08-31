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

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


# Build the waypoint matrix
X = np.array([
        [ 0,  2.0], # p
        [ 0,  0.84], # v
        [ 0,  -4.5], # a
        ])

Y = np.array([
        [ 0,  -2.86], # p
        [ 0,  -1.27], # v
        [ 0,  6.75], # a
        ])

Z = np.array([
        [ 0,  0.05], # p
        [ 0,  -0.23], # v
        [ 0,  -5.86], # a
            ])

W = np.array([
        [ 0,  0.0], # p
        [ 0,  0.0], # v
        [ 0,  0.0], # a
        ])


v_lim = 3.0
a_lim = 14.0

x_cnstr = np.array([[-3.0, 3.0], [-v_lim, v_lim], [-a_lim, a_lim]])
y_cnstr = np.array([[-3.0, 3.0], [-v_lim, v_lim], [-a_lim, a_lim]])
z_cnstr = np.array([[-0.3, 1.0], [-v_lim, v_lim], [-a_lim, a_lim]])
w_cnstr = np.array([[0.0, 1.2],  [-v_lim, v_lim], [-a_lim, a_lim]])

# Generate the polynomial
bz_x = bz.Bezier(waypoints=X, constraints=x_cnstr, degree=18)
bz_y = bz.Bezier(waypoints=Y, constraints=y_cnstr, degree=18)
bz_z = bz.Bezier(waypoints=Z, constraints=z_cnstr, degree=18)
bz_w = bz.Bezier(waypoints=W, constraints=w_cnstr, degree=5)

Curve = bz_t.BezierCurve(bz_x, bz_y, bz_z, bz_w)


print("Evaluation of the bezier polynomial")
print(bz_x.eval(1.0, [0,1,2]))
print(bz_y.eval(1.0, [0,1,2]))
print(bz_z.eval(1.0, [0,1,2]))




#### PLOT
N = 20
test_y = np.zeros((N, 3), dtype=float)
t = np.zeros((N), dtype=float)

for i in range(N):
    t[i] = 1.0/N * i
    test_y[i, :] = bz_x.eval(t[i], [0,1,2])

fig, axs = plt.subplots(3, 1)
axs[0].plot(t, test_y[:, 0], t, np.ones(N) * x_cnstr[0,0], t, np.ones(N) * x_cnstr[0,1])
axs[0].set_title("p")

axs[1].plot(t, test_y[:, 1], t, np.ones(N) * x_cnstr[1,0], t, np.ones(N) * x_cnstr[1,1])
axs[1].set_title("v")

axs[2].plot(t, test_y[:, 2], t, np.ones(N) * x_cnstr[2,0], t, np.ones(N) * x_cnstr[2,1])
axs[2].set_title("a")




