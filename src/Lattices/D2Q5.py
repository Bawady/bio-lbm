import numpy as np
from src.UnitSystem import *
from .Lattice import Lattice


class D2Q5(Lattice):
    D = 2
    Q = 5

    # lattice weights
    __w_rest = 1 / 3.
    __w_others = 1 / 6.
    weights = np.array([__w_rest, __w_others, __w_others, __w_others, __w_others], dtype=float)
    # lattice velocity direction vectors (entries only 0,1)
    dir_x = np.array([0, 1, -1, 0, 0], dtype=int)
    dir_y = np.array([0, 0, 0, -1, 1], dtype=int)
    # Index array for getting the inverse velocity vectors e.g. inverse of [1, 0] at qxys[1] is [-1, 0] at qxys[2]
    qs_n_indices = np.array([0, 2, 1, 4, 3])

    #######################################
    #####          Streaming          #####
    #######################################

    def stream_periodic(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        for i in range(self.Q):
            f_new[i] = np.roll(f_old[i], (mag(self.dir_y[i]), mag(self.dir_x[i])), axis=(0, 1))

    def stream_periodic_neg(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        for i in range(self.Q):
            f_new[i] = np.roll(f_old[i], (-mag(self.dir_y[i]), -mag(self.dir_x[i])), axis=(0, 1))

    def stream(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        height, width = f_new.shape[1], f_new.shape[2]

        f_new[0] = f_old[0]
        f_new[1, :, 1:width]    = f_old[1, :, 0:width-1].copy()
        f_new[2, :, 0:width-1]  = f_old[2, :, 1:width].copy()
        f_new[3, 0:height-1, :] = f_old[3, 1:height, :].copy()
        f_new[4, 1:height, :]   = f_old[4, 0:height-1, :].copy()

    def stream_neg(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        height, width = f_new.shape[1], f_new.shape[2]

        f_new[0] = f_old[0]
        f_new[1, :, 0:width-1]    = f_old[1, :, 1:width].copy()
        f_new[2, :, 1:width]  = f_old[2, :, 0:width-1].copy()
        f_new[3, 1:height, :] = f_old[3, 0:height-1, :].copy()
        f_new[4, 0:height-1, :]   = f_old[4, 1:height, :].copy()

    #######################################
    #####     Boundary Conditions     #####
    #######################################

    def bc_bot(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
               y: int, x0: int, x1: int,
               velocity_profile: np.ndarray = None, density_profile: np.ndarray = None) -> None:
        f[3, y, x0:x1] = f[4, y, x0:x1]

    def bc_top(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
               y: int, x0: int, x1: int,
               velocity_profile: np.array = None, density_profile: np.ndarray = None) -> None:
        f[4, y, x0:x1] = f[3, y, x0:x1]

    def bc_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                x: int, y0: int, y1: int,
                velocity_profile: np.array = None, density_profile: np.array = None) -> None:
        f[1, y0:y1, x] = f[2, y0:y1, x]

    def bc_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                 x: int, y0: int, y1: int,
                 velocity_profile: np.array = None, density_profile: np.array = None) -> None:
        f[2, y0:y1, x] = f[1, y0:y1, x]
