# TODO: define function to load txt files as each two lists (t, value)
## then for x=0 translate each x value by -R and each y value by 0
## and for x=L translate each x value by -R-L and each y value by 0
## maybe use therefore parameter position_on_beam which is basically the mesh.x value and then (x,y) -> (x - mesh_x, y)
## then for x=0 we get theta by new (x,y) with arctan(y / x)
## then for x=L we get (u,v) by inverse of calculate_displacements_in_origin_frame (with theta, x, y given)
## maybe move the transform funcs into transfrom.py (or similar) and ansys.py only load, and simliar to run

import numpy as np
import pandas as pd
from elastic_beam.run import parameters


def load_global_displacement_from_ansys(file_name):
    data = pd.read_csv(
        f"elastic_beam/data/{file_name}.txt",
        sep="\t",
        decimal=","
    ).values
    return data[:, 1:2], data[:, 2:3]


t, x0 = load_global_displacement_from_ansys("ansys_elastic_beam_x=0_global_displacement_x")
_, y0 = load_global_displacement_from_ansys("ansys_elastic_beam_x=0_global_displacement_y")
_, xL = load_global_displacement_from_ansys("ansys_elastic_beam_x=L_global_displacement_x")
_, yL = load_global_displacement_from_ansys("ansys_elastic_beam_x=L_global_displacement_y")


def calculate_displacements_in_rotating_frame(x_displacement, y_displacement, theta_solution, parameters):
    ...
    return ..., ...


x0 += parameters.R
xL += parameters.R + parameters.L
theta = np.arctan(y0 / x0)
x = np.concatenate([x0, xL], axis=1)
y = np.concatenate([y0, yL], axis=1)
# u, v = calculate_displacements_in_rotating_frame(x, y, theta, parameters)

from IPython import embed
embed()
