# main pipeline
from dae_solver import ODE
import numpy as np

# first test with MVG form based on page 275 of book Lynch&Park

ode = lambda theta, theta_dot, theta_dotdot, tau, m1, m2, L1, L2, g: np.array(
    [
        (m1 * L1 ** 2 + m2 * (L1 ** 2 + 2 * L1 * L2 * np.cos(theta[1]) + L2 ** 2)) * theta_dotdot[0]
        + (m2 * (L1 * L2 * np.cos(theta[1]) + L2 ** 2)) * theta_dotdot[1]
        + (-m2 * L1 * L2 * np.sin(theta[1]) * (2 * theta_dot[0] * theta_dot[1] + theta_dot[1] ** 2)) * theta_dot[0]
        + ((m1 + m2) * L1 * g * np.cos(theta[0]) + m2 * g * L2 * np.cos(theta[0] + theta[1])) * theta[0]
        - tau[0],
        (m2 * (L1 * L2 * np.cos(theta[1]) + L2 ** 2)) * theta_dotdot[0]
        + m2 * L2 ** 2 * theta_dotdot[1]
        + (m2 * L1 * L2 * theta_dot[0] ** 2 + np.sin(theta[1])) * theta_dot[1]
        + (m2 * g * L2 * np.cos(theta[0] + theta[1])) * theta[1]
        - tau[1]
    ])

param_ = np.array([1, 1, 1, 1, 1])  # initial (L1, L2, m1, m2)
theta_ = np.array([0, 0, 0])  # initial (theta_1, theta_2, theta_3)
theta_dot_ = np.array([0, 0, 0])  # initial (theta´_1, theta´_2, theta´_3)
theta_dotdot_ = np.array([0, 0, 0])  # initial (theta´´_1, theta´´_2, theta´´_3)
init_cond = [param_, theta_, theta_dot_, theta_dotdot_]
state_array = np.array(["theta", "tau"])
param_array = np.array(["L1", "L2", "m1", "m2", "g"])

mvg_ode = ODE.ODEMODEL(ode=ode,
                       init_cond=init_cond,
                       state_array=state_array,
                       param_array=param_array)
