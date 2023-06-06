import numpy as np
from ufl import SpatialCoordinate, TestFunction, TrialFunction, inner, nabla_grad, nabla_div, dx
from dolfinx.fem import assemble_scalar, form, set_bc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector


class ModelFEMSystem:
    def __init__(self, parameters, mesh, function_spaces):
        self.dim_u = None
        self.dim_v = None
        self.M_u = None
        self.M_v = None
        self.C_uv = None
        self.C_vu = None
        self.K_u = None
        self.K_v = None
        self.K_u1 = None
        self.K_v1 = None
        self.K_v2 = None
        self.K_u2 = None
        self.C_v = None
        self.I_b = None

        self.define_system_forms(parameters, mesh, function_spaces)
        self.assembled = False


    @staticmethod
    def petsc_matrix_to_numpy(mat):
        dim1, dim2 = mat.size
        return np.array([[mat.getValue(i,j) for i in range(dim1)] for j in range(dim2)])


    @staticmethod
    def petsc_vector_to_numpy(vec):
        dim = vec.size
        return np.array([vec.getValue(i) for i in range(dim)])


    def define_system_forms(self, parameters, mesh, function_spaces):
        u = TrialFunction(function_spaces[0])
        v = TrialFunction(function_spaces[1])
        w = TestFunction(function_spaces[0])
        z = TestFunction(function_spaces[1])
        x = SpatialCoordinate(mesh)

        self.M_u = parameters.rho * parameters.A * u * w * dx
        self.M_v = parameters.rho * parameters.A * v * z * dx
        self.C_uv = parameters.rho * parameters.A * u * z * dx
        self.C_vu = parameters.rho * parameters.A * v * w * dx
        self.K_u = parameters.E * parameters.A * inner(nabla_grad(u), nabla_grad(w)) * dx
        self.K_v = parameters.E * parameters.I * nabla_div(nabla_grad(v)) * nabla_div(nabla_grad(z)) * dx
        self.K_u1 = self.M_u
        self.K_v1 = self.M_v
        self.K_v2 = parameters.rho * parameters.A * (
            parameters.R * (parameters.L - x[0]) + 0.5 * (parameters.L + x[0]) * (parameters.L - x[0])
        ) * inner(nabla_grad(v), nabla_grad(z)) * dx

        self.K_u2 = parameters.rho * parameters.A * (parameters.R + x[0]) * w * dx
        self.C_v = parameters.rho * parameters.A * (parameters.R + x[0]) * z * dx

        self.I_b = parameters.rho * parameters.A * (parameters.R + x[0])**2 * dx


    def assemble_system(self, boundary_conditions):
        system_matrices = ["M_u", "M_v", "C_uv", "C_vu", "K_u", "K_v", "K_u1", "K_v1", "K_v2"]
        system_vectors = ["K_u2", "C_v"]
        system_scalars = ["I_b"]

        for idx, value in vars(self).items():
            if idx in system_matrices:
                value = assemble_matrix(form(value), bcs=boundary_conditions)
                value.assemble()
                vars(self)[idx] = self.petsc_matrix_to_numpy(value)
            if idx in system_vectors:
                value = assemble_vector(form(value))
                value.assemble()
                set_bc(value, bcs=boundary_conditions)
                vars(self)[idx] = self.petsc_vector_to_numpy(value)
            if idx in system_scalars:
                vars(self)[idx] = assemble_scalar(form(value))

        self.dim_u = self.M_u.shape[0]
        self.dim_v = self.M_v.shape[0]
        self.assembled = True
