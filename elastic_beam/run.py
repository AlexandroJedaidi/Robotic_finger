import numpy as np

from elastic_beam.parameters import ModelParameters
from elastic_beam.assemble import (
    create_mesh_and_function_spaces,
    define_boundary_conditions,
    get_nodes_for_function_eval,
    calculate_displacements_in_origin_frame
)
from elastic_beam.model import ModelFEMSystem
from elastic_beam.ode import ode_lhs, ode_rhs, ode_solver
from elastic_beam.ansys import (
    load_and_get_displacement_in_origin_frame_from_ansys,
    calculate_displacements_in_rotating_frame
)
from elastic_beam.plot import plot_displacements, animate_displacements


#################### DEFINE MATHEMATICAL AND NUMERICAL PARAMETERS ####################
torque = lambda t: np.piecewise(t, [t < 5 / 3., (10 / 3. > t >= 5 / 3.), t >= 10 / 3.], [2, -2, 0]).item()
# torque = lambda t: 0.25
parameters = ModelParameters(
    num_space_elements=50,
    num_time_steps=150,
    L=1.1,
    T=5,
    rho=2700,
    A=1e-4,
    E=71e9,
    I=8.33e-12,
    R=0.1,
    I_h=3.84,
    theta_initial=0,
    alpha_initial=0,
    tau=torque,
)

if __name__ == "__main__":
    #################### ASSEMBLE FEM SYSTEM ####################
    mesh, function_spaces = create_mesh_and_function_spaces(parameters)
    boundary_conditions = define_boundary_conditions(mesh, function_spaces)
    system = ModelFEMSystem(parameters, mesh, function_spaces)
    system.assemble_system(boundary_conditions)

    #################### SOLVE INITIAL VALUE PROBLEM ####################
    y0 = np.block([
        [np.zeros((system.dim_u + system.dim_v, 1))],
        [np.array(parameters.theta_initial).reshape(1, 1)],
        [np.zeros((system.dim_u + system.dim_v, 1))],
        [np.array(parameters.alpha_initial).reshape(1, 1)],
    ]).reshape(-1,)
    y_solution = ode_solver(
        fun_lhs=ode_lhs,
        fun_rhs=ode_rhs,
        t_span=(0, parameters.T),
        steps=parameters.num_time_steps,
        y0=y0,
        parameters=parameters,
        system=system
    )

    #################### GET SOLUTION AT MESH NODES ####################
    u_solution, v_solution, x_coordinates = get_nodes_for_function_eval(
        y_solution[:, : system.dim_u],
        y_solution[:, system.dim_u : system.dim_u + system.dim_v],
        mesh,
        function_spaces,
        parameters
    )
    theta_solution = y_solution[:, system.dim_u + system.dim_v]

    #################### GET SOLUTION IN CARTESIAN COORDINATES ####################
    x_displacement, y_displacement, x_no_displacement, y_no_displacement = calculate_displacements_in_origin_frame(
        u_solution, v_solution, theta_solution, x_coordinates[:, 0], parameters
    )

    #################### LOAD AND TRANSFORM ANSYS SOLUTION ####################
    x_ansys, y_ansys, theta_ansys, t_ansys = load_and_get_displacement_in_origin_frame_from_ansys(parameters)
    u_ansys, v_ansys = calculate_displacements_in_rotating_frame(x_ansys, y_ansys, theta_ansys)

    #################### PLOT AND ANIMATE SOLUTION ####################
    file_name = "elastic_beam_with_piecewise_torque"
    plot_displacements(
        [u_solution, u_ansys], [v_solution, v_ansys], [theta_solution, theta_ansys], t_ansys, parameters, file_name
    )
    animate_displacements(
        x_displacement, y_displacement, x_no_displacement, y_no_displacement, parameters, file_name
    )

    #################### END OF FILE ####################
    from IPython import embed

    embed()
