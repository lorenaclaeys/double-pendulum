import math
import solver
import graph
import lyapunov
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #### parameters
    # mass (kg)
    m1 = 1.
    m2 = 1.
    # lenght (m)
    l=1.
    #angle (rad)
    th1 = (math.pi)
    th2 = .5*(math.pi)
    dth1 = 0
    dth2 = 0
    #gravity (m/s²)
    g = 9.81
    #constants for the solver
    h = 0.01 #time step
    t_max = 30 #(s)
    #perturbation (rad)
    delta1 = 0.001
    delta2 = 0.001
    ###2nd pendulum
    #angle (rad) & derivatives
    th1A = th1 + delta1
    th2A = th2 + delta2
    dth1A = 0
    dth2A = 0



    ########## 1 double pendulums ##########

    #data from solver
    THETA, T = solver.pendulum(solver.p_derivatives, th1, th2, dth1, dth2, t_max, h, m1, m2, g, l)

    #graph
    ani = graph.grapher(l,THETA, h)
    plt.show()






    ########## 2 double pendulums ##########

    #data from solver
    THETA, T = solver.pendulum(solver.p_derivatives, th1, th2, 0, 0, t_max, h, m1, m2, g, l)
    THETAA, T = solver.pendulum(solver.p_derivatives, th1 + delta1, th2 + delta2, 0, 0, t_max, h, m1, m2, g, l)

    #graph
    ani = graph.grapher_two(l,THETA, THETAA, h)
    plt.show()


    #energy
    H = np.zeros(len(T))
    for (i,t) in enumerate(T):
        H[i] = lyapunov.energy(THETA[i,:], m1, m2, g, l)
    print("The energy is :",H[-1])

    #graph energy
    plt.plot(T,H)
    plt.title("energy")
    plt.xlabel("t(s)")
    plt.ylabel("H(Joules)")
    plt.show()

    #maximal lyapunov exponent
    #initial condition
    delta = THETAA[-1,:] - THETA[-1,:]
    #normalised initial condition
    delta0 = delta/np.linalg.norm(delta)
    #lyapunov exponent
    L=0
    n = 2001
    lyap_max , lyap = lyapunov.lyap(L, n ,delta0 ,delta,THETA, THETAA,h, m1, m2, g, l)
    print("the maximal lyapunov exponent is",lyap_max)

    #graph for lyapunov exponent
    plt.title("Lyapunov exponent")
    plt.xlabel("t(s)")
    plt.ylabel("lyapunov exponent (log)")
    plt.plot(range(n),lyap)
    plt.show()

    #evolution of the lyapunov exponnent in function of the energy
    N = 100
    plt.xlabel("Energy(Joules)")
    plt.ylabel("Lyapunov exponent")
    plt.title("Lyapunov evolution in function of the ernegy")
    Energy, Lyapunov = lyapunov.lyap_evolution(delta1, delta2, delta0, delta, L, n, N, t_max, h, m1, m2, g, l)
    plt.plot(Energy, Lyapunov)
    plt.show()

