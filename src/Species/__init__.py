from src.Factory import GenericFactory
from .FluidParticle import FluidParticle
from .RDEParticle import RDEParticle
from .DeadParticle import DeadParticle
from .DHW24Species import DHW24Bacteria, DHW24Substrate
from .Hilpert05Species import PpG7, Naphtalene
from .Species import Species
from typing import Tuple


class SpeciesFactory(GenericFactory):
    def create(self, name: str, size: Tuple, lattice, params: dict) -> None:
        cls = self._entries.get(name)
        if not cls:
            raise ValueError(f"Factory does not contain an entry with named {name}")
        obj = cls()
        obj.fabricate(size=size, lattice=lattice, params=params)
        obj.init()
        return obj


factory = SpeciesFactory()
factory.add_entry("FluidParticle", FluidParticle)
factory.add_entry("RDEParticle", RDEParticle)
factory.add_entry("DeadParticle", DeadParticle)
factory.add_entry("DHW24Substrate", DHW24Substrate)
factory.add_entry("DHW24Bacteria", DHW24Bacteria)
factory.add_entry("Napthalene", Naphtalene)
factory.add_entry("PpG7", PpG7)
Species.factory = factory