import numpy as np
from src.Serializable import Serializable
from typing import TYPE_CHECKING
from src.Factory import Fabricable
from src.UnitSystem import *

# For circular dependence due to type hints
if TYPE_CHECKING:
    from .Lattice import Lattice


class Lattice(Fabricable, Serializable):
    D = 0
    Q = 0
    weights = None
    # lattice velocity direction vector (entries only 0,1)
    dir_x, dir_y = None, None
    # Index array for getting the inverse velocity vectors
    qs_n_indices = None
    serialize_members = ["dx", "dt", "q", "dir_x", "dir_y", "vel_x", "vel_y", "cs", "cs_n2", "cs_n4"]

    def __init__(self):
        # lattice parameters
        self.dx = Q(0, 'm')
        self.dt = Q(0, 's')
        # lattice velocity magnitude, speed of sound & related constants
        self.q = Q(0, 'm/s')
        self.cs, self.cs_n2, self.cs_n4 = Q(0, 'm/s'), Q(0, 'm/s'), Q(0, 'm/s')

    def fabricate(self, params: dict):
        super().fabricate(params)
        self.q = self.dx / self.dt
        self.vel_x = self.dir_x * self.q
        self.vel_y = self.dir_y * self.q

        # speed of sound (cs) and common expressions using it
        # cs of dx/dt * 1/sqrt(3) is so common that it is used as default here
        self.cs = self.dx / (self.dt * np.sqrt(3))
        self.cs_n2 = self.dt**2 * 3 / self.dx**2
        self.cs_n4 = self.cs_n2**2

    def init(self):
        pass

    #######################################
    #####          Streaming          #####
    #######################################

    def stream_periodic(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        pass

    def stream_periodic_neg(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        pass

    def stream(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        pass

    def stream_neg(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        pass

    #######################################
    #####     Boundary Conditions     #####
    #######################################

    def bc_bot(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
               y: int, x0: int, x1: int,
               velocity_profile: np.ndarray = None, density_profile: np.ndarray = None) -> None:
        pass

    def bc_top(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
               y: int, x0: int, x1: int,
               velocity_profile: np.array = None, density_profile: np.ndarray = None) -> None:
        pass

    def bc_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                x: int, y0: int, y1: int,
                velocity_profile: np.array = None, density_profile: np.array = None) -> None:
        pass

    def bc_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                 x: int, y0: int, y1: int,
                 velocity_profile: np.array = None, density_profile: np.array = None) -> None:
        pass


    ##### Corner Boundaries

    def bc_concave_bot_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                            y: int, x: int) -> None:
        pass

    def bc_concave_bot_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                             y: int, x: int) -> None:
        pass

    def bc_concave_top_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                            y: int, x: int) -> None:
        pass

    def bc_concave_top_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                             y: int, x: int) -> None:
        pass

    def bc_convex_top_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x: int) -> None:
        pass

    def bc_convex_bot_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x: int) -> None:
        pass

    def bc_convex_top_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x: int) -> None:
        pass

    def bc_convex_bot_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x: int) -> None:
        pass
