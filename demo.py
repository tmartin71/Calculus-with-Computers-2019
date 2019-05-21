# Libraries for math and plotting
import numpy as np
import matplotlib as plot
import time

# Plotting utilities
def graph(t, y, true_solution, running_time, h):
    max_error = np.max(np.abs((y - true_solution)))

    plot.plot(t, y)
    plot.plot(t, true_solution, linewidth=4)
    plot.legend(("Appx", "True Solution"),loc="upper left")
    plot.title("Run time = {0}, h = {1}, Max err = {2}".format(running_time, h, max_error))
    plot.show()

def graph_lv(t, y):
    plot.plot(t, y[1:])
    plot.plot(t, y[2:])
    plot.legend(("Prey population", "Predator population"), loc="upper left")
    plot.title("Run time = {0}, h = {1}".format(running_time, h))
    plot.show()

def solve_ode():
    return

if __name__ == "__main__":
    solve_ode()