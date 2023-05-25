'''
Cubic polynoimal Trajectory planning in joint space.
Given - Initial velocity, final velocity and initial time = t0, final time = tf
q(t) = a0 + a1*t + a2*t^2 + a3*t^3 position
qd(t)= a1 + 2*a2*t + 3*a3*t^2  velocity
qdd(t) = 2*a2 + 6 * a3 * t acceleration
q0 - initial position  ( Def * 1)
qf - final position  ( Def * 1)
qd0 - initial angular velocity (dof*1)
qdf -  final angular velocity (dof *1)
m - discrete time steps
We will provide the value of h, q0, qf, qd0, qdf  will be an array
We are considering the time frame t [0:1]
'''
import numpy as np


def cubic_trajectory_plannign(q0, qf, qd0, qdf, m=100):
    n = q0.shape[0]  # check in jupyter note_book # shape of q0 and a0 needs to be same
    # polynomial Parameters

    a0 = np.copy(q0)  # we are copying the value of q0 in a0 making both independent
    a1 = np.copy(qd0)
    a2 = 3 * (qf - q0) - (2 * qd0) - qdf
    a3 = -2 * (qf - q0) + (qdf + qd0)

    # creating the linear time step space

    time_steps = np.linspace(0, 1, num=m)

    # Now we can formulate the putputs we are looking for q - position, qd - ve;ocity, qdd
    # acceleration

    q = np.zeros((n, m))
    qd = np.zeros((n, m))
    qdd = np.zeros((n, m))

    for i in range(len(time_steps)):
        t = time_steps[i]
        t1 = t
        t2 = t * t
        t3 = t * t * t
        q[:, i] = a0 + (a1 * t1) + (a2 * t2) + (a3 * t3)
        qd[:, i] = a1 + (2 * a2 * t1) + (3 * a3 * t2)
        qdd[:, i] = (2 * a2) + 6 * a3 * t1

    return q, qd, qdd

