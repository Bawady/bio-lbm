import numpy as np
from src.UnitSystem import *
from .Lattice import Lattice


class D2Q9(Lattice):
    D = 2
    Q = 9

    # lattice weights
    __w_rest = 4 / 9.
    __w_straight = 1 / 9.
    __w_diag = 1 / 36.
    weights = np.array([__w_rest, __w_straight, __w_straight, __w_straight, __w_straight,
                        __w_diag, __w_diag, __w_diag, __w_diag], dtype=float)
    # lattice velocity direction vectors
    dir_x = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=int)
    dir_y = np.array([0, 0, 0, -1, 1, 1, -1, -1, 1], dtype=int)
    # Index array for getting the inverse velocity vectors e.g.     inverse of [1, 0] at qxys[1] is [-1, 0] at qxys[2]
    qs_n_indices = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])


    #######################################
    #####          Streaming          #####
    #######################################

    def stream_periodic(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        for i in range(self.Q):
            f_new[i] = np.roll(f_old[i], (mag(self.dir_y[i]), mag(self.dir_x[i])), axis=(0, 1))

    def stream_periodic_neg(self, f_new: np.ndarray, f_old: np.ndarray) -> None:
        for i in range(self.Q):
            f_new[i] = np.roll(f_old[i], (-mag(self.dir_y[i]), -mag(self.dir_x[i])), axis=(0, 1))

    @staticmethod
    def stream(f_new: np.ndarray, f_old: np.ndarray) -> None:
            height, width = f_new.shape[1], f_new.shape[2]

            f_new[0] = f_old[0]

            f_new[1, :, 1:width]    = f_old[1, :, 0:width-1].copy()
            f_new[2, :, 0:width-1]  = f_old[2, :, 1:width].copy()
            f_new[3, 0:height-1, :] = f_old[3, 1:height, :].copy()
            f_new[4, 1:height, :]   = f_old[4, 0:height-1, :].copy()

            f_new[5, 1:height, 1:width]     = f_old[5, 0:height-1, 0:width-1].copy()
            f_new[6, 0:height-1, 0:width-1] = f_old[6, 1:height, 1:width].copy()
            f_new[7, 0:height-1, 1:width]   = f_old[7, 1:height, 0:width-1].copy()
            f_new[8, 1:height, 0:width-1]   = f_old[8, 0:height-1, 1:width].copy()

    @staticmethod
    def stream_neg(f_new: np.ndarray, f_old: np.ndarray) -> None:
        height, width = f_new.shape[1], f_new.shape[2]

        f_new[0] = f_old[0]

        f_new[1, :, 0:width-1]    = f_old[1, :, 1:width].copy()
        f_new[2, :, 1:width]  = f_old[2, :, 0:width-1].copy()
        f_new[3, 1:height, :] = f_old[3, 0:height-1, :].copy()
        f_new[4, 0:height-1, :]   = f_old[4, 1:height, :].copy()

        f_new[5, 0:height-1, 0:width-1]     = f_old[5, 1:height, 1:width].copy()
        f_new[6, 1:height, 1:width] = f_old[6, 0:height-1, 0:width-1].copy()
        f_new[7, 1:height, 0:width-1]   = f_old[7, 0:height-1, 1:width].copy()
        f_new[8, 0:height-1, 1:width]   = f_old[8, 1:height, 0:width-1].copy()

    #######################################
    #####     Boundary Conditions     #####
    #######################################

    # Auxiliary functions to determine the density of a straight boundary segment
    def __density_bot(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x0: int,
                      x1: int) -> None:
        scale = self.q / (self.q + fluid_vel[1, y, x0:x1])
        density[y, x0:x1] = scale * (f[0, y, x0:x1] + f[1, y, x0:x1] + f[2, y, x0:x1] +
                                     2 * (f[4, y, x0:x1] + f[5, y, x0:x1] + f[8, y, x0:x1]))

    def __density_top(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x0: int,
                      x1: int) -> None:
        scale = self.q / (self.q - fluid_vel[1, y, x0:x1])
        density[y, x0:x1] = scale * (f[0, y, x0:x1] + f[1, y, x0:x1] + f[2, y, x0:x1] +
                                     2 * (f[3, y, x0:x1] + f[6, y, x0:x1] + f[7, y, x0:x1]))

    def __density_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y0: int, y1: int,
                       x: int) -> None:
        scale = self.q / (self.q - fluid_vel[0, y0:y1, x])
        density[y0:y1, x] = scale * (f[0, y0:y1, x] + f[3, y0:y1, x] + f[4, y0:y1, x] +
                                     2 * (f[2, y0:y1, x] + f[6, y0:y1, x] + f[8, y0:y1, x]))

    def __density_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y0: int, y1: int, x) -> None:
        scale = self.q / (self.q + fluid_vel[0, y0:y1, x])
        density[y0:y1, x] = scale * (f[0, y0:y1, x] + f[3, y0:y1, x] + f[4, y0:y1, x] + 2 * (
            f[1, y0:y1, x] + f[5, y0:y1, x] + f[7, y0:y1, x]))

    # Wet-node non-equilibrium bounce-back scheme for straight wall boundaries
    def bc_bot(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
               y: int, x0: int, x1: int,
               velocity_profile: np.ndarray = None, density_profile: np.ndarray = None) -> None:
        if density_profile is None:
            self.__density_bot(f, density, fluid_vel, y, x0, x1)
            fluid_vel[:, y, x0:x1] = velocity_profile
        elif velocity_profile is None:
            density[y, x0:x1] = density_profile
            fluid_vel[0, y, x0:x1] = Q(0, unit(fluid_vel))
            fluid_vel[1, y, x0:x1] = self.q * ((f[0, y, x0:x1] + f[1, y, x0:x1] + f[2, y, x0:x1] + 2 * (f[4, y, x0:x1] + f[5, y, x0:x1] + f[8, y, x0:x1])) / density_profile - 1)
        else:
            raise ValueError("Neither velocity nor density profile given")

        wall_dens = density[y, x0:x1]
        vel_x = fluid_vel[0, y, x0:x1]
        vel_y = fluid_vel[1, y, x0:x1]
        # up
        f[3, y, x0:x1] = f[4, y, x0:x1] - 2 * wall_dens * vel_y / (3 * self.q)
        # up-left
        f[6, y, x0:x1] = (f[5, y, x0:x1] + 0.5 * (f[1, y, x0:x1] - f[2, y, x0:x1])
                          + wall_dens / self.q * (-0.5 * vel_x - 1 / 6. * vel_y))
        # up-right
        f[7, y, x0:x1] = (f[8, y, x0:x1] - 0.5 * (f[1, y, x0:x1] - f[2, y, x0:x1])
                          + wall_dens / self.q * (0.5 * vel_x - 1 / 6. * vel_y))

    # Wet-node non-equilibrium bounce-back scheme for top
    def bc_top(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
               y: int, x0: int, x1: int,
               velocity_profile: np.array = None, density_profile: np.ndarray = None) -> None:
        if density_profile is None:
            self.__density_top(f, density, fluid_vel, y, x0, x1)
            fluid_vel[:, y, x0:x1] = velocity_profile
        elif velocity_profile is None:
            density[y, x0:x1] = density_profile
            fluid_vel[0, y, x0:x1] = 0
            fluid_vel[1, y, x0:x1] = self.q * (1 - (f[0, y, x0:x1] + f[1, y, x0:x1] + f[2, y, x0:x1] +
                                                    2 * (f[3, y, x0:x1] + f[6, y, x0:x1] + f[7, y, x0:x1])) / density_profile)
        else:
            raise ValueError("Neither velocity nor density profile given")

        wall_dens = density[y, x0:x1]
        vel_x = fluid_vel[0, y, x0:x1]
        vel_y = fluid_vel[1, y, x0:x1]
        # down
        f[4, y, x0:x1] = f[3, y, x0:x1] + 2 * wall_dens * vel_y / (3 * self.q)
        # down-right
        f[5, y, x0:x1] = (f[6, y, x0:x1] - 0.5 * (f[1, y, x0:x1] - f[2, y, x0:x1])
                          + wall_dens / self.q * (0.5 * vel_x + 1 / 6. * vel_y))
        # down-left
        f[8, y, x0:x1] = (f[7, y, x0:x1] + 0.5 * (f[1, y, x0:x1] - f[2, y, x0:x1])
                          + wall_dens / self.q * (-0.5 * vel_x + 1 / 6. * vel_y))


    # Wet-node non-equilibrium bounce-back scheme for bottom
    def bc_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                x: int, y0: int, y1: int,
                velocity_profile: np.array = None, density_profile: np.array = None) -> None:
        if density_profile is None:
            self.__density_left(f, density, fluid_vel, y0, y1, x)
            fluid_vel[:, y0:y1, x] = velocity_profile
        elif velocity_profile is None:
            density[y0:y1, x] = density_profile
            fluid_vel[1, y0:y1, x] = 0
            fluid_vel[0, y0:y1, x] = self.q * (1 - (f[0, y0:y1, x] + f[3, y0:y1, x] + f[4, y0:y1, x] +
                                                    2 * (f[2, y0:y1, x] + f[6, y0:y1, x] + f[8, y0:y1, x])) / density_profile)
        else:
            raise ValueError("Neither velocity nor density profile given")

        wall_dens = density[y0:y1, x]
        vel_x = fluid_vel[0, y0:y1, x]
        vel_y = fluid_vel[1, y0:y1, x]
        # right
        f[1, y0:y1, x] = f[2, y0:y1, x] + 2 * wall_dens * vel_x / (3 * self.q)
        # down-right
        f[5, y0:y1, x] = (f[6, y0:y1, x] - 0.5 * (f[4, y0:y1, x] - f[3, y0:y1, x])
                          + wall_dens / self.q * (1 / 6. * vel_x + 0.5 * vel_y))
        # up-right
        f[7, y0:y1, x] = (f[8, y0:y1, x] + 0.5 * (f[4, y0:y1, x] - f[3, y0:y1, x])
                          + wall_dens / self.q * (1 / 6. * vel_x - 0.5 * vel_y))

    def bc_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                 x: int, y0: int, y1: int,
                 velocity_profile: np.array = None, density_profile: np.array = None) -> None:
        if density_profile is None:
            self.__density_right(f, density, fluid_vel, y0, y1, x)
            fluid_vel[:, y0:y1, x] = velocity_profile
        elif velocity_profile is None:
            density[y0:y1, x] = density_profile
            fluid_vel[1, y0:y1, x] = Q(0, 'm/s')
            fluid_vel[0, y0:y1, x] = self.q * ((f[0, y0:y1, x] + f[3, y0:y1, x] + f[4, y0:y1, x] +
                                                2 * (f[1, y0:y1, x] + f[5, y0:y1, x] + f[7, y0:y1, x])) / density_profile - 1)
        else:
            raise ValueError("Neither velocity nor density profile given")

        wall_dens = density[y0:y1, x]
        vel_x = fluid_vel[0, y0:y1, x]
        vel_y = fluid_vel[1, y0:y1, x]
        # down
        f[2, y0:y1, x] = f[1, y0:y1, x] - 2 * wall_dens * vel_x / (3 * self.q)
        # down-right
        f[6, y0:y1, x] = (f[5, y0:y1, x] + 0.5 * (f[4, y0:y1, x] - f[3, y0:y1, x])
                          - wall_dens / self.q * (1 / 6. * vel_x + 0.5 * vel_y))
        # down-left
        f[8, y0:y1, x] = (f[7, y0:y1, x] - 0.5 * (f[4, y0:y1, x] - f[3, y0:y1, x])
                          + wall_dens / self.q * (-1 / 6. * vel_x + 0.5 * vel_y))

    ##### Corner Boundaries

    # For concave corners multiple populations are undefined
    # The two with a velocity perpendicular to the edge facing inwards the respective corners are arbitrary,
    # as long as they are of the same magnitude
    def bc_concave_bot_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                            y: int, x: int) -> None:
        # impose known macroscopic values (from neighbors)
        fluid_vel[0, y, x] = fluid_vel[0, y, x + 1]
        fluid_vel[1, y, x] = fluid_vel[1, y - 1, x]
        density[y, x] = density[y, x + 1]

        f[1, y, x] = f[2, y, x] + 2. * density[y, x] * fluid_vel[0, y, x] / (3 * self.q)
        f[3, y, x] = f[4, y, x] - 2. * density[y, x] * fluid_vel[1, y, x] / (3 * self.q)
        f[7, y, x] = f[8, y, x] + density[y, x] * (fluid_vel[0, y, x] - fluid_vel[1, y, x]) / (6 * self.q)
        f[5, y, x] = Q(0., unit(f))
        f[6, y, x] = Q(0., unit(f))
        f[0, y, x] = density[y, x] - np.sum(f[1:9, y, x])

    def bc_concave_bot_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                             y: int, x: int) -> None:
        fluid_vel[0, y, x] = fluid_vel[0, y, x - 1]
        fluid_vel[1, y, x] = fluid_vel[1, y - 1, x]
        density[y, x] = density[y, x - 1]

        f[2, y, x] = f[1, y, x] - 2. * density[y, x] * fluid_vel[0, y, x] / (3 * self.q)
        f[3, y, x] = f[4, y, x] - 2. * density[y, x] * fluid_vel[1, y, x] / (3 * self.q)
        f[6, y, x] = f[5, y, x] - density[y, x] * (fluid_vel[0, y, x] + fluid_vel[1, y, x]) / (6 * self.q)
        f[7, y, x] = Q(0, unit(f))
        f[8, y, x] = Q(0, unit(f))
        f[0, y, x] = density[y, x] - np.sum(f[1:9, y, x])

    def bc_concave_top_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                            y: int, x: int) -> None:
        fluid_vel[0, y, x] = fluid_vel[0, x, x + 1]
        fluid_vel[1, y, x] = fluid_vel[1, y + 1, y]
        density[y, x] = density[y, x + 1]

        f[1, y, x] = f[2, y, x] + 2. * density[y, x] * fluid_vel[0, y, x] / (3 * self.q)
        f[4, y, x] = f[3, y, x] + 2. * density[y, x] * fluid_vel[1, y, x] / (3 * self.q)
        f[5, y, x] = f[6, y, x] + density[y, x] * (fluid_vel[0, y, x] + fluid_vel[1, y, x]) / (6 * self.q)
        f[7, y, x] = Q(0, unit(f))
        f[8, y, x] = Q(0, unit(f))
        f[0, y, x] = density[y, x] - np.sum(f[1:9, y, x])

    def bc_concave_top_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray,
                             y: int, x: int) -> None:
        fluid_vel[0, y, x] = fluid_vel[0, y, x - 1]
        fluid_vel[1, y, x] = fluid_vel[1, y + 1, y]
        density[y, x] = density[y, x - 1]

        f[2, y, x] = f[1, y, x] - 2. * density[y, x] * fluid_vel[0, y, x] / (3 * self.q)
        f[4, y, x] = f[3, y, x] + 2. * density[y, x] * fluid_vel[1, y, x] / (3 * self.q)
        f[8, y, x] = f[7, y, x] + density[y, x] * (-fluid_vel[0, y, x] + fluid_vel[1, y, x]) / (6 * self.q)
        f[5, y, x] = Q(0, unit(f))
        f[6, y, x] = Q(0, unit(f))
        f[0, y, x] = density[y, x] - np.sum(f[1:9, y, x])

    # For convex corners only the velocity facing inwards is undefined
    def bc_convex_top_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x: int) -> None:
        f[7, y, x] = f[8, y, x]

    def bc_convex_bot_right(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x: int) -> None:
        f[5, y, x] = f[6, y, x]

    def bc_convex_top_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x: int) -> None:
        f[6, y, x] = f[5, y, x]

    def bc_convex_bot_left(self, f: np.ndarray, density: np.ndarray, fluid_vel: np.ndarray, y: int, x: int) -> None:
        f[8, y, x] = f[7, y, x]
