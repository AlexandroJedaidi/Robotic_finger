import numpy as np
from Calculating_M_matrix import M_matrix

th1 = np.pi/6
th2 = np.pi/3
th1dot = 0
th2dot = 1
Mat = M_matrix(th1, th2, th1dot, th2dot)
print(Mat)
