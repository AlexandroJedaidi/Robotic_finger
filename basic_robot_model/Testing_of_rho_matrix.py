import numpy as np
from Calculating_rho_matrix import rho_matrix

th1 = np.pi/6
th2 = np.pi/3
th1dot = 3
th2dot = 1

rho_m = rho_matrix(th1, th2, th1dot, th2dot)
print(rho_m)
