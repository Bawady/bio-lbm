from src.MultiLBM import MultiLBM
from src.UnitSystem import *
from utils.dump_util import *
from utils.geomtry_util import *


class PostStamp(MultiLBM):
    def init_sim_params(self, params: dict) -> None:
        self.x = Quantity(23, 'm')
        self.y = Quantity(23, 'm')
        self.dx = Quantity(1, 'm')
        self.dt = Quantity(1, 's')
        self.corner_dens_mult = 2.

    def calculate_sim_params(self) -> None:
        self.add_species("BasicParticle", "Species", viscosity=Quantity(0.129, "m**2/s"))

    def boundaries(self) -> None:
        self.default_boundaries("Species")

    def init_geometry(self) -> None:
        solid_rectangle(0, self.width, 0, self.height, self.solid)

    def init_state(self) -> None:
        species = self.get_species("Species")
        solid_circle_filled(0, 0, 5, species.density, self.corner_dens_mult)
        solid_circle_filled(0, self.height-1, 5, species.density, self.corner_dens_mult)
        solid_circle_filled(self.width-1, 0, 5, species.density, self.corner_dens_mult)
        solid_circle_filled(self.width-1, self.height-1, 5, species.density, self.corner_dens_mult)
        species.init_populations(self.bulk_velocity)

    def dump(self, directory: str) -> None:
        dump_img(self, directory, get_density, species="Species")
        dump_img(self, directory, get_fluid_vel, select="xy", cmap='RdBu', species="Species")
        self.dump_count += 1
