from numpy import sin,cos
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import solver

######graph for 1 pendulum
def grapher(l, THETA, h):
    """function that animates the motions equations for a double pendulum"""
    #cartesian coordinates
    x1 = l * np.sin(THETA[:,0])
    y1 = -l * np.cos(THETA[:,0])
    x2 = x1 + l * np.sin(THETA[:,1])
    y2 = y1 - l * np.cos(THETA[:,1])

    fig = plt.figure()
    ax = fig.add_subplot(111,autoscale_on = False, xlim = (-2,2), ylim = (-2,2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init():
        """function that initiates the animation"""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        """function that fill the init function with the values of the motion equation of the double pendulum"""
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*h))
        return line, time_text

    anim = animation.FuncAnimation(fig, animate, range(1, len(THETA)), interval=h*1000, blit=True, init_func=init)
    plt.plot(x1,y1, linewidth = 0.1)
    plt.plot(x2,y2, linewidth = 0.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Single double pendulum")
    plt.show()
    return anim


######graph for 2 double pendulums
def grapher_two(l, THETA, THETAA, h):
    """function that animates the motions equations for two double pendulums with the possibility to put 2 differents initial conditions  """
    #cartesian coordinates
    x1 = l * np.sin(THETA[:,0])
    y1 = -l * np.cos(THETA[:,0])
    x2 = x1 + l * np.sin(THETA[:,1])
    y2 = y1 - l * np.cos(THETA[:,1])

    x1A = l * np.sin(THETAA[:,0])
    y1A = -l * np.cos(THETAA[:,0])
    x2A = x1A + l * np.sin(THETAA[:,1])
    y2A = y1A - l * np.cos(THETAA[:,1])

    fig = plt.figure()
    ax = fig.add_subplot(111,autoscale_on = False, xlim = (-2,2), ylim = (-2,2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init():
        """function that initiates the animation"""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        """function that fill the init function with the values of the motion equation of the two double pendulums"""
        thisx = [x2[i], x1[i], 0, x1A[i], x2A[i]]
        thisy = [y2[i], y1[i], 0, y1A[i], y2A[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*h))
        return line, time_text

    anim_two = animation.FuncAnimation(fig, animate, range(1, len(THETA)), interval=h*1000, blit=True, init_func=init)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Pair of double pendulum")
    plt.show()
    return anim_two

