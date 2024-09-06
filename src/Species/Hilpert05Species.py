import numpy as np
from .Species import Species
from src.Lattices import Lattice
from src.UnitSystem import *


class Naphtalene(Species):
    serialize_members = ["diffusivity", "bgk_relax", "tau", "diffusivity", "neigh_dens"]
    diffusivity = Q(7.5E-10, 'm**2/s')
    kd = DimQ(2.1E-3, 'g/L')

    def init(self) -> None:
        self.diffusivity = Q(Naphtalene.diffusivity)
        self.tau = self.diffusivity * self.lattice.cs_n2 + self.lattice.dt / 2
        self.bgk_relax = 1 / self.tau
        self.neigh_dens = Q(np.zeros(self.f.shape, dtype=float), unit(self.density))

    def update_feq(self, bulk_velocity: np.ndarray = None) -> None:
        self.update_feq_lin(bulk_velocity)

    def update_moments(self) -> None:
        self.density = np.sum(self.f, axis=0)

    def stream_dens(self) -> None:
        for i in range(self.lattice.Q):
            self.neigh_dens[i] = np.roll(self.density, (-self.lattice.dir_y[i], -self.lattice.dir_x[i]), axis=(0, 1))
#        height, width = self.size
#
#        self.neigh_dens[1, :, 0:width-1]    = self.density[:, 1:width]
#        self.neigh_dens[2, :, 1:width]  = self.density[:, 0:width-1]
#        self.neigh_dens[3, 1:height, :] = self.density[0:height-1, :]
#        self.neigh_dens[4, 0:height-1, :]   = self.density[1:height, :]
#        self.neigh_dens[5, 0:height-1, 0:width-1]     = self.density[1:height, 1:width]
#        self.neigh_dens[6, 1:height, 1:width] = self.density[0:height-1, 0:width-1]
#        self.neigh_dens[7, 1:height, 0:width-1]   = self.density[0:height-1, 1:width]
#        self.neigh_dens[8, 0:height-1, 1:width]   = self.density[1:height, 0:width-1]

    def density_gradient(self):
        self.stream_dens()

        grad = Q(np.zeros((2, *self.size)), 'g/(m**3*m)')
        cnt = 0
        for i in range(1, self.lattice.Q):
            if self.lattice.vel_x[i] != 0:
                grad[0] += self.neigh_dens[i] / (self.lattice.vel_x[i] * self.lattice.dt)
                cnt += 1
            if self.lattice.vel_y[i] != 0:
                grad[1] += self.neigh_dens[i] / (self.lattice.vel_y[i] * self.lattice.dt)
        grad[0, :, 0] *= 0
        grad[0, :, -1] *= 0
        grad[1, 0, :] *= 0
        grad[1, -1, :] *= 0
        return grad / cnt


# Pseudmonas Putida G7
class PpG7(Species):
    serialize_members = ["motility", "bgk_relax", "tau", "vc", "xi", "swim_speed"]
    motility = DimQ(3.2E-11, 'm**2/s')
    xi = DimQ(1.8E-9, 'm**2/s')
    swim_speed = DimQ(48E-6, 'm/s')

    def init(self) -> None:
        self.motility = Q(PpG7.motility)
        self.xi = Q(PpG7.xi)
        self.swim_speed = Q(PpG7.swim_speed)
        self.vc = q_zeros_like(self.velocity)
        self.tau = self.motility * self.lattice.cs_n2 + self.lattice.dt / 2
        self.bgk_relax = 1 / self.tau

    def update_feq(self, bulk_velocity: np.ndarray = None) -> None:
        self.__update_chem_velocity()
        if bulk_velocity is not None:
            self.update_feq_lin(self.vc + bulk_velocity)
        else:
            self.update_feq_lin(self.vc)

    def update_moments(self) -> None:
        self.density = np.sum(self.f, axis=0)

    def __update_chem_velocity(self):
        grad = self.substrate.density_gradient()
        grad_abs = np.sqrt(grad[0]**2 + grad[1]**2)
        kd = Q(Naphtalene.kd)
        tanh_arg = self.xi / (2 * self.swim_speed) * kd / (kd + self.substrate.density)**2 * grad_abs
        grad_abs[mag(grad_abs) == 0] = Q(1, unit(grad_abs))
        assert np.min(tanh_arg) >= 0
        self.vc = 2 / 3 * self.swim_speed * np.tanh(mag(tanh_arg)) * grad / grad_abs
