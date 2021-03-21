import math
import solver
import graph
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

    #gravity
    g = 9.81
    #constants for the solver
    h = 0.01
    t_max = 30


    THETA, T = solver.pendulum(solver.p_derivatives, th1, th2, dth1, dth2, t_max, h, m1, m2, g, l)

    ani = graph.grapher(l, THETA, h)
    plt.show()
