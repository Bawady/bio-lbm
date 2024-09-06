import numpy as np
import pint

from src.MultiLBM import MultiLBM
from src.UnitSystem import *
from utils.dump_util import *
from utils.geomtry_util import *
from test import *
import matplotlib.pyplot as plt


# Reproduce results of De Rosis, Harish and Wang in "Lattices Boltzmann modelling of bacterial colony patterns"
class DHW24(MultiLBM):
    def setup(self) -> None:
        # Directly use the non-dimensional quantities reported by the authors
        set_conversion_mode(ConversionMode.NO_PINT)
        self.x, self.y = 500, 500
        self.dx = 1
        self.dt = 1
        self.react_params["cap_at_zero"] = True

        alpha1, alpha2 = 1. / 2400., 1. / 120.
        tau_sub = 2.5

        sub = self.add_species("DHW24Substrate", "substrate", "D2Q5",
                         tau=tau_sub, init_quantity=self.density_sub)
        bac = self.add_species("DHW24Bacteria", "bacteria", "D2Q5",
                         diffusivity=self.diff_bac * sub.diffusivity, init_quantity=0)
        dead = self.add_species("DeadParticle", "dead", "D2Q5", init_quantity=0)

        self.add_reaction([1 * bac, 0 * sub], [1 * dead], self.death_rate, alpha1=alpha1, alpha2=alpha2)
        self.add_reaction([1 * bac, 1 * sub], [2 * bac], self.growth_rate)

    def init_state(self) -> None:
        density_bac = 1
        self.bacteria.density[self.height // 2, self.width // 2] = density_bac

    @staticmethod
    def death_rate(reactants, externals, products, alpha1, alpha2):
        bac, sub = reactants[0], externals[0]
        return 1 / ((1 + sub.density / alpha2) * (1 + bac.density / alpha1))

    @staticmethod
    def growth_rate(reactants, externals, products):
        return 1

    def dump(self, directory: str) -> None:
        cnt = str(self.dump_count)
        dump_as_img(self.substrate.density, directory, "sub_density_" + cnt, cmap='viridis')
        dump_as_img(self.bacteria.density, directory, "bac_density_" + cnt, cmap='viridis')
        dump_as_img(self.bacteria.density + self.dead.density, directory, "total_density_" + cnt, cmap='viridis')
        self.dump_count += 1


if __name__ == '__main__':
    configs = [("disk", {"diff_bac": 0.25, "density_sub": 0.25}),
               ("dbm", {"diff_bac": 0.12, "density_sub": 0.071}),
               ("dla", {"diff_bac": 0.05, "density_sub": 0.087}),
               ("ring", {"diff_bac": 0.05, "density_sub": 0.1})]
    sel_config = 2
    sim = DHW24()
    sim.fabricate(configs[sel_config][1])
    sim = MultiLBM.load_checkpoint(f"sim_out/DHW24/{configs[sel_config][0]}/chkpt")
    # To "continue" a simulation, just comment-out the above three lines and uncomment the one below
#    sim = MultiLBM.deserialize(f"sim_out/DHW24/{configs[sel_config][0]}/chkpt")
    sim.run(5000, 250, f"sim_out/DHW24/{configs[sel_config][0]}")
    sim.serialize(f"sim_out/DHW24/{configs[sel_config][0]}/chkpt")
