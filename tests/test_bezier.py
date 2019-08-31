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

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


# Build the waypoint matrix
X = np.array([
        [ 0,  2.0], # p
        [ 0,  0.84], # v
        [ 0,  -4.5], # a
        ])

x_cnstr = np.array([[-3.0, 3.0], [-2.0, 2.0], [-14.0, 14.0]])

# Generate the polynomial
bz_x = bz.Bezier(waypoints=X, constraints=x_cnstr, degree=7)

print("Evaluation of the bezier polynomial")
print(bz_x.eval(1.0, [0,1,2]))



#### PLOT
N = 50
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




