"""
Bezier Polynomial
@author: l.pannocchi@gmail.com
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import numpy as np
import numpy.matlib 

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

class BZ5deg:
    def __init__(self, Initial, Final, Lb, Ub):

        self.P0 = Initial[0]
        self.V0 = Initial[1]
        self.A0 = Initial[2]

        self.Pf = Final[0]
        self.Vf = Final[1]
        self.Af = Final[2]

        self.Lb = Lb
        self.Ub = Ub

        self.NPCPoints = 6
        self.NVCPoints = 5
        self.NACPoints = 4

        self.P = np.zeros(self.NPCPoints)
        self.V = np.zeros(self.NVCPoints)
        self.A = np.zeros(self.NACPoints)

        self.Aconstr = list()
        self.bconstr = list() 
        
        self.current_t = 1.0

        # Initially evaluate the control points at t = 1.0
        (self.P, self.V, self.A) = self.evalCPoints(self.current_t)


    def updateInitial(self, Initial):
        """
        Update the initial conditions and the control points
        """
        self.P0 = Initial[0]
        self.V0 = Initial[1]
        self.A0 = Initial[2]

        (self.P, self.V, self.A) = self.evalCPoints(self.current_t)


    def updateFinal(self, Final):
        """
        Update the initial conditions and the control points
        """
        self.Pf = Final[0]
        self.Vf = Final[1]
        self.Af = Final[2]

        (self.P, self.V, self.A) = self.evalCPoints(self.current_t)


    def evalCPoints(self, t): 
        self.current_t = t;
        
        P = np.zeros(self.NPCPoints)
        V = np.zeros(self.NVCPoints) 
        A = np.zeros(self.NACPoints)

        P[self.NPCPoints - 1] = self.Pf # P[5] 
        V[self.NVCPoints - 1] = self.Vf # V[4]
        A[self.NACPoints - 1] = self.Af # A[3]
        P[0] = self.P0
        V[0] = self.V0
        A[0] = self.A0

        V[1] = V[0]
        V[3] = V[self.NVCPoints - 1] - t * A[self.NACPoints - 1] / 4.0 

        P[1] = t * V[0] / 5.0 + P[0]
        P[2] = P[1] + t * V[1] / 5.0
        P[4] = P[self.NPCPoints - 1] - t * V[self.NVCPoints - 1] / 5.0
        P[3] = P[4] - t * V[3] / 5.0 

        V[2] = (P[3] - P[2]) / t * 5.0

        A[1] = (V[2] - V[1]) / t * 4.0 
        A[2] = (V[3] - V[2]) / t * 4.0

        return (P, V, A)


    def checkFeas(self, t, Initial = None, Final = None):
        output = True 
      
        A_local = np.zeros(self.NACPoints)
        V_local = np.zeros(self.NVCPoints)

        if (Initial is not None):
            updateInitial(Initial)

        if (Final is not None):
            self.updateFinal(Final)
        
        (_, V_local, A_local) = self.evalCPoints(t)

        if (np.isnan(A_local).any()):
            print("Found NAN")
            print(A)
            print("t = ")
            print(t)


        for el in np.nditer(A_local):
            if (el > self.Ub or el < self.Lb):
                return (False, [min(A_local), max(A_local)])
        
        return (True, [min(A_local), max(A_local)])
    
