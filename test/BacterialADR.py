import numpy as np
import pint

from src.MultiLBM import MultiLBM
from src.UnitSystem import *
from utils.dump_util import *
from utils.geomtry_util import *
import matplotlib.pyplot as plt


def get_total_bac(sim: "MultiLBM", params: dict):
    return "density_total", sim.bacteria.density + sim.dead.density


class BacterialADR(MultiLBM):
    def setup(self) -> None:
        set_conversion_mode(ConversionMode.NO_PINT)
        self.x, self.y = 500, 500
        self.dx = 1
        self.dt = 1

        alpha1, alpha2 = 1. / 2400., 1. / 120.
        tau_sub = 2.5
        density_sub = 0.071
        diff_bac = 0.08

#        fluid = self.add_species("BasicParticle", "fluid", "D2Q9",
#                               viscosity=0.129, init_quantity=1)
        sub = self.add_species("DHW24Substrate", "substrate", "D2Q5",
                               tau=tau_sub, init_quantity=density_sub)
        bac = self.add_species("DHW24Bacteria", "bacteria", "D2Q5",
                               diffusivity=diff_bac * sub.diffusivity, init_quantity=0)
        dead = self.add_species("DeadParticle", "dead", "D2Q5", init_quantity=0)
        sub.update_feq = sub.update_feq_lin
        bac.update_feq = bac.update_feq_lin

        self.add_reaction([1 * bac, 0 * sub], [1 * dead], self.death_rate, alpha1=alpha1, alpha2=alpha2)
        self.add_reaction([1 * bac, 1 * sub], [2 * bac], self.growth_rate)


        # Generating 2D grids 'x' and 'y' using meshgrid with 10 evenly spaced points from -1 to 1
        x, y = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))
        d = np.sqrt(x*x + y*y)
        sigma, mu = 1.25, 0.0
        g = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))

        self.bulk_velocity = np.load("test/sim_out/LidDrivenCavity/CenterVortex/all/results/species/velocity.npy")
        dump_as_img(g, "test/sim_out/BacterialADR/reproduce/", "gauss", cmap='viridis')
        dump_as_img(np.sqrt(self.bulk_velocity[0]**2 + self.bulk_velocity[1]**2), "test/sim_out/BacterialADR/reproduce/", "bv", cmap='viridis')
        test = self.bulk_velocity * g
        dump_as_img(np.sqrt(test[0]**2 + test[1]**2), "test/sim_out/BacterialADR/reproduce/", "test", cmap='viridis')

    def init_state(self) -> None:
        self.bacteria.density[self.height // 2, self.width // 2] = 1

    @staticmethod
    def death_rate(reactants, externals, products, alpha1, alpha2):
        bac = reactants[0]
        sub = externals[0]
        return 1 / ((1 + sub.density / alpha2) * (1 + bac.density / alpha1))

    @staticmethod
    def growth_rate(reactants, externals, products):
        return 1

    def boundaries(self) -> None:
        velocity = np.zeros((2, self.width))
        velocity[0] = 0.4 * (1 - math.exp(-self.runs**2 / (2 * 1E04)))
#        self.wall_boundary(self.fluid, 0, 0, self.width, self.fluid.lattice.bc_top, velocity_profile=velocity)

    def dump(self, directory: str) -> None:
        cnt = str(self.dump_count)
        dump_as_img(self.substrate.density, directory, "sub_density_" + cnt, cmap='viridis')
        dump_as_img(self.bacteria.density, directory, "bac_density_" + cnt, cmap='viridis')
        dump_as_img(self.bacteria.density + self.dead.density, directory, "total_density_" + cnt, cmap='viridis')
#        dump_as_img(np.sqrt(self.fluid.velocity[0]**2 + self.fluid.velocity[1]**2), directory,
#                    "fluid_velocity_" + cnt, cmap='viridis')

        path = directory + "/species" + str(self.dump_count) + ".png"
        rgb_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        w, h, c = rgb_data.shape
        for iy in range(h):
            for ix in range(w):
                rgb_data[iy, ix, 0] = int(self.substrate.density[iy, ix] * 8 * 255)
                rgb_data[iy, ix, 1] = int(self.bacteria.density[iy, ix] * 8 * 255)
                rgb_data[iy, ix, 2] = int(self.dead.density[iy, ix] * 8 * 255)
        plt_img.imsave(path, rgb_data, dpi=600)

        self.dump_count += 1
