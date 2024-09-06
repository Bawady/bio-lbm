import pint

from src.MultiLBM import MultiLBM
from src.UnitSystem import *
from utils.dump_util import *
from utils.geomtry_util import *
import matplotlib.pyplot as plt


# Reproduce paper by Qian and Orszag about simulation of RDE A+B->C
# Collision: BGK-based with additional term
# Equilibrium: Uniform
class QO94(MultiLBM):
    def setup(self) -> None:
        self.dx, self.dt, self.initial_density = characteristics(Q(1, 'm'), Q(1, 's'), Q(1, 'kg/m**3'))

        self.x = Q(201, 'm')
        self.y = Q(201, 'm')
        self.dx = Q(1, 'm')
        self.dt = Q(1, 's')
        self.viscosity = Q(1/2, "m**2/s")
        self.density_a = Q(1., 'kg/m**3')
        self.density_b = Q(1., 'kg/m**3')
        self.u_max = Q(0.4, 'm/s')

        self.react_params["flavor"] = 2

        speca = self.add_species("RDEParticle", "speca", "D2Q5", init_quantity=Q(0, 'kg/m**3'), viscosity=self.viscosity)
        specb = self.add_species("RDEParticle", "specb", "D2Q5", init_quantity=Q(0, 'kg/m**3'), viscosity=self.viscosity)
        specc = self.add_species("RDEParticle", "specc", "D2Q5", init_quantity=Q(0, 'kg/m**3'), viscosity=self.viscosity)
        self.add_reaction([1 * speca, 1 * specb], [1 * specc], self.reaction_rate)

        self.prod_rate = q_zeros_like(specc.density)

        self.fig, self.axs = plt.subplots(2,)
        self.fig.suptitle("center domain velocity")
        self.axs[0].set_ylim(0, 1.2)
        self.axs[1].legend(loc="upper right")

    @staticmethod
    def reaction_rate(reactants, externals, products):
        return Q(0.01, 'm**3/(kg * s)')

    def init_geometry(self) -> None:
        solid_rectangle(0, self.width, 0, self.height, self.solid)

    def init_state(self) -> None:
        filled_rectangle(0, self.width // 2, 0, self.height, self.speca.density, self.density_a)
        filled_rectangle(self.width // 2, self.width, 0, self.height, self.specb.density, self.density_b)

    def boundaries(self) -> None:
        for species in [self.speca, self.specb]:
            self.wall_boundary(species, 0, 0, self.width,
                               species.lattice.bc_top, velocity_profile=Q(0, 'm/s'))
            self.wall_boundary(species, -1, 0, self.height,
                               species.lattice.bc_right, velocity_profile=Q(0, 'm/s'))
            self.wall_boundary(species, -1, 0, self.width,
                               species.lattice.bc_bot, velocity_profile=Q(0, 'm/s'))
            self.wall_boundary(species, 0, 0, self.height,
                               species.lattice.bc_left, velocity_profile=Q(0, 'm/s'))

    def dump(self, directory: str) -> None:
        if self.runs == 0:
            return
        cnt = str(self.dump_count)
        self.axs[0].plot(self.speca.density[self.height // 2, :], 'r', label="A")
        self.axs[0].plot(self.specb.density[self.height // 2, :], 'b', label="B")
        self.axs[1].plot(self.prod_rate[self.height // 2, :], 'g', label=f"rate{self.runs}")

        if not directory.endswith("/"):
            directory += "/"
        plt.savefig(f"{directory}post_sim{cnt}.png")

        self.dump_count += 1

    def collide(self) -> None:
        super().collide()
        self.prod_rate = np.sum(self.specc.f_coll, axis=0)

    def stream(self) -> None:
        self.prod_rate = np.sum(self.specc.f_coll, axis=0) - self.prod_rate
        super().stream()

    def post_sim(self, show: bool, store_at: str = "") -> None:
        plt.tight_layout()
        if not store_at == "":
            if not store_at.endswith("/"):
                store_at += "/"
            plt.savefig(store_at + "post_sim.png")
        if show:
            plt.show()


if __name__ == '__main__':
    sim = QO94()
    sim.run(1200, 200, "sim_out/QO94")
    sim.post_sim(show=True, store_at="sim_out/QO94")