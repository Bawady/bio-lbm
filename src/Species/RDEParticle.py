import numpy as np

from .Species import Species
from src.Lattices import Lattice


class RDEParticle(Species):
    serialize_members = ["viscosity", "tau", "bgk_relax"]

    def init(self) -> None:
        self.tau = self.lattice.dt / 2 + self.viscosity * self.lattice.cs_n2
        self.bgk_relax = (self.lattice.dt / self.tau)

    def update_feq(self, bulk_velocity: np.ndarray = None) -> None:
        return self.update_feq_lin(bulk_velocity)

    def update_moments(self) -> None:
        self.density = np.sum(self.f, axis=0)
