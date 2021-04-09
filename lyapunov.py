import solver
import graph
import math
import matplotlib.pyplot as plt
from math import log
import numpy as np

######## parameters
#gravity(m/s²)
g = 9.81
#constants for the solver
h = 0.01 #time step
t_max = 20 #(s)
#masses (kg)
m1 = 1.
m2 = 1.
# lenght (m)
l=1.
###1st pendulum
#angle & derivatives (rad)
th1 = .5*(math.pi)
th2 = .0*(math.pi)
dth1 = 0
dth2 = 0
#perturbation (rad)
delta1 = 0.001
delta2 = 0.001
###2nd pendulum
#angle (rad) & derivatives
th1A = th1 + delta1
th2A = th2 + delta2
dth1A = 0
dth2A = 0

#data from solver
THETA, T = solver.pendulum(solver.p_derivatives, th1, th2, 0, 0, t_max, h, m1, m2, g, l)
THETAA, T = solver.pendulum(solver.p_derivatives, th1 + delta1, th2 + delta2, 0, 0, t_max, h, m1, m2, g, l)


# plt.plot(n,THETA[:,0]%2*math.pi)
#plt.plot(T,THETAA[:,0]%2*math.pi)
#plt.plot(T,THETA[:,1]%2*math.pi)
#plt.plot(T,THETAA[:,1]%2*math.pi)
# plt.show()



########energy
def energy(theta, m1, m2, g, l):
    th1 = theta[0]
    th2 = theta[1]
    dth1 = theta[2]
    dth2 = theta[3]
    V = -(m1+m2)*l*g*math.cos(th1) - m2*l*g*math.cos(th2)
    K = .5*m1*(l*dth1)**2 + .5*m2*((l*dth1)**2 + (l*dth2)**2 + 2*l*l*dth1*dth2*math.cos(th1-th2))
    return K + V

#graph for energy
H = np.zeros(len(T))
for (i,t) in enumerate(T):
    H[i] = energy(THETA[i,:], m1, m2, g, l)
print(H[-1])
plt.plot(T,H)
plt.title("energy")
plt.xlabel("t")
plt.ylabel("H")
plt.show()

#graph two double pendulums
ani = graph.grapher_two(l,THETA, THETAA, h)
plt.show()



########lyapunov exponent
#initial condition
delta = THETAA[-1,:] - THETA[-1,:]
#normalised initial condition
delta0 = delta/np.linalg.norm(delta)
#lyapunov exponent
L=0
n = 2001
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
lambda_max = L/n/h #
print("the maximal lyapunov exponent is",lambda_max)
#graph
lyap2 = lyap
plt.title("lyapunov exponent")
plt.xlabel("t")
plt.ylabel("lyapunov exponent")
plt.plot(range(n),lyap2)
plt.show()






#question 6

th1= np.linspace(1,20,20)
H_tot = np.zeros(int(20/1))
H = np.zeros(len(T))
for k in range(0,20,1): #1 to 10
    #data from solver
    THETA, T = solver.pendulum(solver.p_derivatives, th1[k], th2, 0, 0, t_max, h, m1, m2, g, l)
    THETAA, T = solver.pendulum(solver.p_derivatives, th1[k] + delta1, th2 + delta2, 0, 0, t_max, h, m1, m2, g, l)
    for (i,t) in enumerate(T):
        H[i] = energy(THETA[i,:], m1, m2, g, l)
    H_tot[k] = H[0]

lambda_max_tot= np.zeros(20)
for k in range(0,20,1): #1 to 10
    THETA, T = solver.pendulum(solver.p_derivatives, th1[k], th2, 0, 0, t_max, h, m1, m2, g, l)
    THETAA, T = solver.pendulum(solver.p_derivatives, th1[k] + delta1, th2 + delta2, 0, 0, t_max, h, m1, m2, g, l)
    ########lyapunov exponent
    #initial condition
    delta = THETAA[-1,:] - THETA[-1,:]
    #normalised initial condition
    delta0 = delta/np.linalg.norm(delta)
    #lyapunov exponent
    L=0
    n = 2001
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
    lambda_max_tot[k]= lambda_max
print("h",H_tot,"l",lambda_max_tot)

plt.plot(H_tot,lambda_max_tot)
plt.show()
