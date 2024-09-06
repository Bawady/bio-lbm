import numpy as np
from .Species import Species
from src.Lattices import Lattice


class DHW24Substrate(Species):
    serialize_members = ["diffusivity", "bgk_relax"]

    def init(self) -> None:
        self.diffusivity = (self.tau - 0.5) * self.lattice.cs**2
        self.bgk_relax = (self.lattice.dt / self.tau)

    def update_feq(self, bulk_velocity: np.ndarray = None) -> None:
        self.update_feq_rest()

    def update_moments(self) -> None:
        self.density = np.sum(self.f, axis=0)


class DHW24Bacteria(Species):
    serialize_members = ["diffusivity", "bgk_relax"]

    def init(self) -> None:
        self.tau = self.diffusivity * self.lattice.cs_n2 + 0.5
        self.bgk_relax = 1 / self.tau

    def update_feq(self, bulk_velocity: np.ndarray = None) -> None:
        self.update_feq_rest()

    def update_moments(self) -> None:
        self.density = np.sum(self.f, axis=0)
