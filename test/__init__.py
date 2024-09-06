from src.Factory import GenericFactory
from src.MultiLBM import MultiLBM

from .Couette import Couette
from .Poiseuille import Poiseuille
from .LidDrivenCavity import LidDrivenCavity
from .QO94 import QO94
from .DHW24 import DHW24
from .PHUP21 import PHUP21
from .Hilpert05 import Hilpert05


factory = GenericFactory()
factory.add_entry("Couette", Couette)
factory.add_entry("Poiseuille", Poiseuille)
factory.add_entry("LidDrivenCavity", LidDrivenCavity)
factory.add_entry("QO94", QO94)
factory.add_entry("DHW24", DHW24)
factory.add_entry("PHUP21", PHUP21)
factory.add_entry("Hilpert05", Hilpert05)
MultiLBM.factory = factory
