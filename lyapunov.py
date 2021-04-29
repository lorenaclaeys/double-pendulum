import solver
import graph
import math
import matplotlib.pyplot as plt
from math import log
import numpy as np


#trajectories
#plt.plot(n,THETA[:,0]%2*math.pi)
#plt.plot(T,THETAA[:,0]%2*math.pi)
#plt.plot(T,THETA[:,1]%2*math.pi)
#plt.plot(T,THETAA[:,1]%2*math.pi)
#plt.show()



#energy
def energy(theta, m1, m2, g, l):
    """function that returns the energy of the double pendulum"""
    th1 = theta[0]
    th2 = theta[1]
    dth1 = theta[2]
    dth2 = theta[3]
    V = -(m1+m2)*l*g*math.cos(th1) - m2*l*g*math.cos(th2)
    K = .5*m1*(l*dth1)**2 + .5*m2*((l*dth1)**2 + (l*dth2)**2 + 2*l*l*dth1*dth2*math.cos(th1-th2))
    return K + V




#lyapunov exponent
def lyap(L, n ,delta0 ,delta,THETA, THETAA, h, m1, m2, g, l):
    """fuction that returns the maximal lyapunov exponent and a vector containing the evolution of the lyapunov coefficient""" 
    lyap = np.zeros(n)
    i=0
    while i < n:
        THETAA, _ = solver.pendulum(solver.p_derivatives, THETA[-1, 0]+delta0[0], THETA[-1, 1]+delta0[1], THETA[-1, 2]+delta0[2], THETA[-1, 3]+delta0[3], h, h, m1, m2, g, l)
        THETA, _ = solver.pendulum(solver.p_derivatives, THETA[-1, 0], THETA[-1, 1], THETA[-1, 2], THETA[-1, 3], h, h, m1, m2, g, l)
        #renormalisation
        delta = THETAA[-1,:] - THETA[-1,:]
        norm = np.linalg.norm(delta)
        L += log(norm)
        delta0 = delta/norm
        i += 1
        lyap[i-1] = L/i/h
    #maximal lyapunov exponent (log)
    lambda_max = L/n/h
    return lambda_max, lyap


#lyapunov evolution
Lyapunov = []
Energy = []
def lyap_evolution(delta1, delta2, delta0, delta, L, n, N, t_max, h, m1, m2, g, l):
    """function that gives the evolution of the maximal lyapunov coefficient in function of the energy"""
    for k in range(N):
        #we make our energy vary in function of theta1
        th1 = (1*math.pi)/ (N-k)
        th2 = 0
        THETA, T = solver.pendulum(solver.p_derivatives, th1, th2, 0, 0, t_max, h, m1, m2, g, l)
        THETAA, T = solver.pendulum(solver.p_derivatives, th1 + delta1, th2 + delta2, 0, 0, t_max, h, m1, m2, g, l)
        H = np.zeros(len(T))
        for (i,t) in enumerate(T):
            H[i] = energy(THETA[i,:], m1, m2, g, l)
        H_average = np.amin(H)
        Energy.append(H_average)
        lambda_max,_ = lyap(L, n ,delta0 ,delta,THETA, THETAA, h, m1, m2, g, l)
        Lyapunov.append(lambda_max)
    return Energy, Lyapunov

