import numpy as np

from .Species import Species
from src.Lattices import Lattice


class FluidParticle(Species):
    serialize_members = ["viscosity", "tau", "bgk_relax"]

    def init(self) -> None:
        self.tau = self.lattice.dt / 2 + self.viscosity * self.lattice.cs_n2
        self.bgk_relax = (self.lattice.dt / self.tau)
