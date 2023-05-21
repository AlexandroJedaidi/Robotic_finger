from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class ModelParameters:

    num_space_elements: int
    num_time_steps: int
    L: float                    # beam length
    T: float                    # end time point
    rho: float                  # density
    A = float                   # cross-sectional area
    E = float                   # Young's modules
    I = float                   # area moment of inertia
    R = float                   # radius of rigid hub
    I_h = float                 # mass moment of inertia of hub
    theta_initial: float        # initial rotation position of hub beam system
    alpha_initial: float        # initial rotation velocity of hub beam system
    tau: Callable               # supplied torque from actuator


@dataclass
class ModelSystem:
    dim_u: int
    dim_v: int

    M_u: Any
    M_v: Any
    C_uv: Any
    C_vu: Any
    K_u: Any
    K_v: Any
    K_u1: Any
    K_v1: Any
    K_v2: Any

    K_u2: Any
    C_v: Any

    I_b: Any

