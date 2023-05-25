from casadi import *
import numpy as np


class DAE:

    def __init__(self, ode, init_cond, state_array, param_array, algebr_func, algebr_state):
        self.ode = ode
        self.init_cond = init_cond
        self.state_array = state_array
        self.param_array = param_array
        self.algebr_func = algebr_func
        self.algebr_state = algebr_state
        self.initialize_casadi()

    def initialize_casadi(self):
        # initialize init_cond into separate variables
        self.P_ = ...
        self.X_ = ...
        self.Z_ = ...
        # initialize dae dictionary for integrator
        dae = {}
        self.f = Function()

    def checks(self):
        # checking initial condition
        res = self.f(p=self.P_, x=self.X_, z=self.Z_)
        # TODO: check res["ode"] == XDOT with np.testing

        # checking jacobian
        J = self.f.factory("J", self.f.name_in(), ["jac:alg:z"])

        return False

    def print_problem(self):
        print("ode: ", self.ode)
        print("initial conditions: ", self.init_cond)
        print("algebraic functions: ", self.algebr_func)
        print("algebraic states: ", self.algebr_state)
