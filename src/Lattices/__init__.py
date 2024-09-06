from src.Factory import GenericFactory
from .D2Q9 import D2Q9
from .D2Q5 import D2Q5
from .Lattice import Lattice

factory = GenericFactory()
factory.add_entry("D2Q9", D2Q9)
factory.add_entry("D2Q5", D2Q5)
Lattice.factory = factory