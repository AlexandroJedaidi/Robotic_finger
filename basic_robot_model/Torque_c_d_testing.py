import numpy as np
from Torque_calculations import system_torque

th1 = np.pi/6
th2 = np.pi/3
a1 = 3
a2 = 3
th1dot = 1
th2dot = 2
m1 = 5
m2 = 5
x1 = 3
y1 = 3
z1 = 3
x2 = 3
y2 = 3
z2 = 3
th1ddot = 4
th2ddot = 5
ixx1 = 3
iyy1 = 4
izz1 = 4
ixy1 = 4
ixz1 = 4
iyz1 = 4
ixx2 = 4
iyy2 = 4
izz2 = 4
ixy2 = 5
ixz2 = 5
iyz2 = 6
g = 9.8

torque_1, torque_2 = system_torque(th1, th2, a1, a2, th1dot, th2dot, m1, m2, x1, y1, z1, x2, y2, z2, th1ddot,
                                   th2ddot, ixx1, iyy1, izz1, ixy1, ixz1, iyz1, ixx2, iyy2, izz2, ixy2, ixz2, iyz2, g=9.8)

print(torque_1, torque_2)