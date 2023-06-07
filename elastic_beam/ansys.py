import numpy as np
import pandas as pd


def load_global_displacement_from_ansys(file_name):
    data = pd.read_csv(
        f"elastic_beam/data/{file_name}.txt",
        sep="\t",
        decimal=","
    ).values
    return data[:, 1:2], data[:, 2:3]


def load_and_get_displacement_in_origin_frame_from_ansys(parameters):
    t, x0 = load_global_displacement_from_ansys("ansys_elastic_beam_x=0_global_displacement_x")
    _, y0 = load_global_displacement_from_ansys("ansys_elastic_beam_x=0_global_displacement_y")
    _, xL = load_global_displacement_from_ansys("ansys_elastic_beam_x=L_global_displacement_x")
    _, yL = load_global_displacement_from_ansys("ansys_elastic_beam_x=L_global_displacement_y")

    x0 += parameters.R
    xL += parameters.R + parameters.L

    theta = np.arctan(y0 / x0)
    x = np.concatenate([x0, xL], axis=1)
    y = np.concatenate([y0, yL], axis=1)
    return x, y, theta, t


def calculate_displacements_in_rotating_frame(x_displacement, y_displacement, theta_solution):
    u_solution, v_solution = np.zeros_like(x_displacement), np.zeros_like(y_displacement)
    for i in range(u_solution.shape[0]):
        u_solution[i] = x_displacement[i] * np.cos(theta_solution[i]) \
                            + y_displacement[i] * np.sin(theta_solution[i]) \
                            - x_displacement[0]
        v_solution[i] = -x_displacement[i] * np.sin(theta_solution[i]) \
                            + y_displacement[i] * np.cos(theta_solution[i])
    return u_solution, v_solution
