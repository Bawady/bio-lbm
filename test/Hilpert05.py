import numpy as np
import time
import matplotlib.pyplot as plt

from src.MultiLBM import MultiLBM
from src.UnitSystem import *
from utils.dump_util import *
from utils.geomtry_util import *


# Reproduce paper by Hilpert about chemotaxis moving in a solute
class Hilpert05(MultiLBM):
    def setup(self) -> None:
        self.dx, _, self.sub_init, self.bac_init = characteristics(Q(0.1, 'mm'),
                                                                   Species.PpG7.swim_speed,
                                                                   Q(2.83E-2, 'kg/m**3'),
                                                                   Q(4E12, 'cfu/m**3'))
        self.x = Q(10, 'mm')
        self.y = Q(10, 'mm')
        self.dt = self.dx / Q(Species.PpG7.swim_speed)

        self.react_params["flavor"] = 1

        self.ks = Q(1.3E-4, 'g/L')
        self.q = Q(7.9E-16, 'g/(cfu*s)')

        sub = self.add_species("Napthalene", "substrate", "D2Q9",
                                init_quantity=self.sub_init)
        bac = self.add_species("PpG7", "bacteria", "D2Q9",
                                init_quantity=Q(0, 'cfu/m**3'), substrate=sub)

        self.add_reaction([1 * bac, 1 * sub], [1 * bac], self.sub_consumption, q=self.q, ks=self.ks, dt=sub.lattice.dt)

    @staticmethod
    def sub_consumption(reactants, externals, products, q, ks, dt):
        sub = reactants[1]
        return q / (sub.density + ks)

    def init_state(self) -> None:
        solid_circle_filled(self.width//2-1, self.height//2-1, int(non_dim(Q(1, 'mm'))),
                            self.bacteria.density, self.bac_init)

    def dump(self, directory: str) -> None:
        fig = plt.figure()
        vc = np.sqrt(self.bacteria.vc[0] ** 2 + self.bacteria.vc[1] ** 2)

        fig.suptitle(f"Concentrations after {time.strftime('  %H:%M:%S', time.gmtime(mag(self.time(unit='s'))))}")
        ax = fig.add_subplot(2, 3, 1)
        ax.plot(self.substrate.density[self.height // 2, :], 'b')
        ax.title.set_text("Substrate")
        ax = fig.add_subplot(2, 3, 2)
        ax.plot(dim(vc[self.height // 2, :], 'um/s'), 'b')
        ax.title.set_text("Chem. Vel")
        ax = fig.add_subplot(2, 3, 3)
        ax.plot(self.bacteria.density[self.height // 2, :], 'b')
        ax.title.set_text("Bacteria")

        x = np.arange(0, self.width)
        y = np.arange(0, self.height)
        x, y = np.meshgrid(x, y)
        ax = fig.add_subplot(2, 3, 4, projection='3d')
        ax.plot_surface(x, y, self.substrate.density, cmap=plt.cm.coolwarm)
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        ax.plot_surface(x, y, dim(vc, 'um/s'), cmap=plt.cm.coolwarm)
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        ax.plot_surface(x, y, self.bacteria.density, cmap=plt.cm.coolwarm)

        plt.tight_layout()

        if not directory == "":
            if not directory.endswith("/"):
                directory += "/"
            plt.savefig(directory + f"dump_{self.dump_count}.png", bbox_inches='tight')
        plt.close()
        self.dump_count += 1


if __name__ == '__main__':
    sim = Hilpert05()
    sim.run(8000, 400, "sim_out/Hilpert05/")
