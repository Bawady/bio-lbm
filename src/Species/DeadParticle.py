import numpy as np
from .Species import Species
from src.Lattices import Lattice


class DeadParticle(Species):

    def update_feq(self, bulk_velocity: np.ndarray = None) -> None:
        pass

    def collide_self(self) -> None:
        pass

    def update_moments(self) -> None:
        self.density = np.sum(self.f_coll, axis=0)

    def stream(self) -> None:
        pass
