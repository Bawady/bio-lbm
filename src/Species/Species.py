import numpy as np
from src.Serializable import Serializable
from src.Factory import Fabricable
from src.Lattices import Lattice
from src.UnitSystem import *


class Species(Fabricable, Serializable):
    serialize_members = ["f", "feq", "f_coll", "density", "velocity", "size", "lattice", "bgk_relax", "init_quantity", "name"]

    def __init__(self):
        self.lattice = None
        self.f = None
        self.feq = None
        self.f_coll = None
        self.density = None
        self.velocity = None
        self.bgk_relax = None
        self.size = None
        self.init_quantity = None
        self.name = ""

    def fabricate(self, size: tuple[int, ...], lattice: Lattice, params: dict) -> None:
        super().fabricate(params)
        self.size = size
        self.lattice = lattice
        self.init_quantity = self.init_quantity if hasattr(self, 'init_quantity') else Q(1, 'kg/m**3')
        self.f = Q(np.zeros((self.lattice.Q, *self.size), dtype=float), unit(self.init_quantity))
        self.feq = Q(np.zeros((self.lattice.Q, *self.size), dtype=float), unit(self.init_quantity))
        self.f_coll = Q(np.zeros((self.lattice.Q, *self.size), dtype=float), unit(self.init_quantity))
        self.density = np.ones(self.size, dtype=float) * self.init_quantity
        self.velocity = Q(np.zeros((self.lattice.D, *self.size), dtype=float), 'm/s')

    def init_populations(self, bulk_velocity: np.ndarray = None) -> None:
        self.update_feq(bulk_velocity)
        self.f = self.feq.copy()
        self.f_coll = self.feq.copy()

    def update_moments(self) -> None:
        self.density = np.sum(self.f, axis=0)
        self.velocity[0] = self.lattice.q * np.sum(self.lattice.dir_x[:, None, None] * self.f, axis=0) / self.density
        self.velocity[1] = self.lattice.q * np.sum(self.lattice.dir_y[:, None, None] * self.f, axis=0) / self.density

    def density_mask(self, mask: np.ndarray) -> None:
        self.density *= mask

    def collide_self(self) -> None:
        relax = self.lattice.dt * self.bgk_relax
        self.f_coll = (1 - relax) * self.f + relax * self.feq

    def update_feq_rest(self) -> None:
        for i in range(self.lattice.Q):
            self.feq[i] = self.lattice.weights[i] * self.density

    def update_feq_lin(self, bulk_velocity: np.ndarray = None) -> None:
        fluid_vel = bulk_velocity if bulk_velocity is not None else self.velocity
        cs_n2 = self.lattice.cs_n2
        vel_x = self.lattice.vel_x
        vel_y = self.lattice.vel_y
        ws = self.lattice.weights
        for i in range(self.lattice.Q):
            lin_term = cs_n2 * (vel_x[i] * fluid_vel[0] + vel_y[i] * fluid_vel[1])
            self.feq[i] = ws[i] * self.density * (1 + lin_term)

    def update_feq_sq(self, bulk_velocity: np.ndarray = None) -> None:
        fluid_vel = bulk_velocity if bulk_velocity is not None else self.velocity
        cs_n2 = self.lattice.cs_n2
        vel_x = self.lattice.vel_x
        vel_y = self.lattice.vel_y
        ws = self.lattice.weights
        u_sq = cs_n2 / 2. * (fluid_vel[0] ** 2 + fluid_vel[1] ** 2)
        for i in range(self.lattice.Q):
            lin_term = cs_n2 * (vel_x[i] * fluid_vel[0] + vel_y[i] * fluid_vel[1])
            self.feq[i] = ws[i] * self.density * (1 + lin_term + lin_term ** 2 / 2.0 - u_sq)

    def update_feq(self, bulk_velocity: np.ndarray = None) -> None:
        self.update_feq_sq(bulk_velocity)

    def stream(self) -> None:
        self.lattice.stream(self.f, self.f_coll)

    def stream_periodic(self) -> None:
        self.lattice.stream_periodic(self.f, self.f_coll)

    def density_gradient(self):
        pass

    def __rmul__(self, other):
        return self, other

    def __mul__(self, other):
        return self, other
