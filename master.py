# Libraries for math and plotting
import numpy as np
import matplotlib.pyplot as plot
import time

def y_prime(t, y):
    return -15*y;

def lotka_volterra(t, y):

    # Caribou growth rate
    alpha = 1.1

    # Caribou death rate
    beta = 0.4

    # Wolf death rate
    gamma = 0.4

    # Wolf growth rate
    delta = 0.1

    dx = alpha * y[0] - beta * y[1]*y[0]
    dy = delta * y[0] * y[1] - gamma * y[1]

    return np.array([dx, dy])

def true_solution(t, y):
    return np.exp(-15*t)

def euler_step(h, t, y_n, y_prime):
    return y_n + h * y_prime(t, y_n)

def trapezoid_step(h, t, y_n, y_prime):
    y_tilde = y_n + h*y_prime(t, y_n)
    return y_n + (h / 2) * (y_prime(t, y_n) + y_prime(t + h, y_tilde))

def rk_45_step(h, t, y_n, y_prime):
    t_next = t + h
    k1 = h*y_prime(t, y_n)
    k2 = h*y_prime(t + h/2, y_n + k1/2)
    k3 = h*y_prime(t + h/2, y_n + k2/2)
    k4 = h*y_prime(t + h, y_n + k3)

    return y_n + 1.0/6.0 * (k1 + 2*k2 + 2*k3 + k4)

def graph(t, y, true_solution, running_time, h):
    max_error = np.max(np.abs((y - true_solution)))
    plot.plot(t, y)
    plot.plot(t, true_solution, linewidth=4)
    plot.legend(("Appx", "True Solution"),loc="upper left")
    plot.title("Run time = {0}, h = {1}, Max err = {2}".format(running_time, h, max_error))
    plot.show()

def graph_lv(t, y, running_time, h):
    plot.plot(t, y[0,:])
    plot.plot(t, y[1,:])
    plot.legend(("Prey population", "Predator population"), loc="upper left")
    plot.title("Run time = {0}, h = {1}".format(running_time, h))
    plot.show()

def solve_ode_1d():

    # Domain setup
    h = 0.1
    t_max = 1

    # Initial conditions
    t_0 = 0
    y_0 = 1

    # Linearly spaced vector of length int(t_max / h)
    num_steps = int((t_max - t_0) / h)
    t = np.linspace(t_0, t_max, num_steps)
    
    # Solution array
    y = np.zeros(num_steps)

    # Initial condition
    y[0] = y_0

    start_time = time.time()
    for i in range(0, num_steps - 1):
        y[i + 1] = rk_45_step(h, t[i], y[i], y_prime)
    end_time = time.time()

    running_time = end_time - start_time
    y_true = true_solution(t, y)
    graph(t, y, y_true, running_time, h)

def solve_ode_nd():

    # Domain setup
    h = 0.1
    t_max = 100

    # Initial conditions
    t_0 = 0
    y_0 = np.array([10, 10])

    # Linearly spaced vector of length int(t_max / h)
    num_steps = int((t_max - t_0) / h)
    t = np.linspace(t_0, t_max, num_steps)
    
    # Solution array
    y = np.zeros((2, num_steps))

    # Initial condition
    y[:,0] = y_0

    start_time = time.time()
    for i in range(0, num_steps - 1):
        y[:, i + 1] = rk_45_step(h, t[i], y[:,i], lotka_volterra)
    end_time = time.time()

    running_time = end_time - start_time
    graph_lv(t, y, running_time, h)

if __name__ == "__main__":
    solve_ode_nd()
    #main_loop()