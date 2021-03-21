import solver
import graph
import math
import matplotlib.pyplot as plt
from math import log
import numpy as np

#gravity
g = 9.81
#constants for the solver
h = 0.01
t_max = 120
#parametres
m1 = 1.
m2 = 1.
# lenght (m)
l=1.
###1st pendulum
#angle (rad)
th1 = (math.pi)
th2 = .5*(math.pi)
dth1 = 0
dth2 = 0
#perturbation
delta1 = 0.001
delta2 = 0.001
###2nd pendulum
th1A = th1 + delta1
th2A = th2 + delta2
dth1A = 0
dth2A = 0


THETA, T = solver.pendulum(solver.p_derivatives, th1, th2, 0, 0, t_max, h, m1, m2, g, l)
THETAA, T = solver.pendulum(solver.p_derivatives, th1 + delta1, th2 + delta2, 0, 0, t_max, h, m1, m2, g, l)

#graph
ani = graph.grapher_lyap(l,THETA, THETAA, h)
plt.show()


###lyapunov exponent
delta = THETAA[-1,:] - THETA[-1,:]
delta0 = delta/np.linalg.norm(delta)
L=0
n=10

lyap = np.zeros(n)
i=0
while i < n:
    THETAA, T = solver.pendulum(solver.p_derivatives, THETA[-1, 0]+delta0[0], THETA[-1, 1]+delta0[1], THETA[-1, 2]+delta0[2], THETA[-1, 3]+delta0[3], t_max, h, m1, m2, g, l)
    THETA, T = solver.pendulum(solver.p_derivatives, THETA[-1, 0], THETA[-1, 1], THETA[-1, 2], THETA[-1, 3], t_max, h, m1, m2, g, l)
    delta = THETAA[-1,:] - THETA[-1,:]
    norm = np.linalg.norm(delta)
    L += log(norm)
    delta0 = delta/norm
    lyap[i] = L
    i += 1


lambda_max = L/(t_max*n) #attention, donné en log
print(lambda_max)
lyap_max = lyap/(t_max*n)
t = [0,1,2,3,4,5,6,7,8,9]
plt.plot(t,lyap_max) 
plt.show()



###energy
def energy(theta, m1, m2, g, l):
    th1 = theta[0]
    th2 = theta[1]
    dth1 = theta[2]
    dth2 = theta[3]
    V = -(m1+m2)*l*g*math.cos(th1) - m2*l*g*math.cos(th2)
    T = .5*m1*(l*dth1)**2 + .5*m2*((l*dth1)**2 + (l*dth2)**2 + 2*l*l*dth1*dth2*math.cos(th1-th2))
    return T + V

H = np.zeros(len(T))
for (i,t) in enumerate(T):
    H[i] = energy(THETA[i,:], m1, m2, g, l)
plt.plot(T,H)
plt.show()
