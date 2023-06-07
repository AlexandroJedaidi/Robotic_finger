import numpy as np
from Calculating_rho_matrix import rho_matrix
from Trajectory_Testing import q, qd
R_t = np.empty((len(q[0]), 2, 2))
print(q.shape)
for i in range(len(q[0])):
    th1 = q[0][i]
    th2 = q[1][i]
    th1dot = qd[0][i]
    th2dot = qd[1][i]
    R_t[i] = rho_matrix(th1, th2, th1dot, th2dot)
# print(R_t)


