from src.MultiLBM import MultiLBM
from src.UnitSystem import *
import matplotlib.pyplot as plt
from utils.dump_util import *
from utils.geomtry_util import *
from test import *


class LidDrivenCavity(MultiLBM):
    serialize_members = ["reynolds", "u_max"]

    def setup(self) -> None:
        self.x, self.dt, density = characteristics(Q(100, 'm'), Q(1, 's'), Q(1, 'kg/m**3'))
        self.reynolds = 100
        self.init_density = Q(1, 'kg/m**3')
        nodes = 128
        self.dx = self.x / nodes
        self.y = Q(100, 'm')
        self.u_max = Q(0.1, 'm/s')
        viscosity = self.u_max * self.x / self.reynolds
        self.sigma = math.floor(10 * self.x)

        self.add_species("FluidParticle", "species", "D2Q9", init_quantity=self.init_density,
                         viscosity=viscosity)
        print(self.reynolds)
        print(self.species.tau)

    def init_geometry(self) -> None:
        solid_rectangle(0, self.width, 0, self.height, self.solid)

    def wall_velocity(self, size: int, dir: int) -> np.ndarray:
        velocity = Q(np.zeros((2, size)), 'm/s')
        velocity[dir] = self.u_max * (1 - math.exp(-self.runs**2 / (2 * self.sigma**2)))
        return velocity

    def boundaries(self) -> None:
#        if self.walls[0] == 1:
        self.wall_boundary(self.species, -1, 0, self.width,
                           self.species.lattice.bc_bot, velocity_profile=self.wall_velocity(self.width, 0))
        self.wall_boundary(self.species, 0, 0, self.height,
                           self.species.lattice.bc_left, velocity_profile=0)
        self.wall_boundary(self.species, -1, 0, self.height,
                           self.species.lattice.bc_right, velocity_profile=0)
        self.wall_boundary(self.species, 0, 0, self.width,
                           self.species.lattice.bc_top, velocity_profile=0)

        self.corner_boundary(self.species, 0, 0, self.species.lattice.bc_concave_top_left)
        self.corner_boundary(self.species, -1, 0, self.species.lattice.bc_concave_bot_left)
        self.corner_boundary(self.species, -1, -1, self.species.lattice.bc_concave_bot_right)
        self.corner_boundary(self.species, 0, -1, self.species.lattice.bc_concave_top_right)

    def dump(self, directory: str) -> None:
        vel = self.species.velocity
        dump_as_img(np.sqrt(vel[0]**2 + vel[1]**2), directory, f"velocity_{self.dump_count}")
        self.dump_count += 1

    def post_sim(self, show: bool = False, store_at: str = "") -> None:
        if show or not store_at == "":
            x_ind = np.arange(self.width) / self.width
            xdata = self.species.velocity[1, self.height // 2, :] / self.u_max
            y_ind = np.arange(self.height) / self.height
            ydata = self.species.velocity[0, :, self.width // 2] / self.u_max

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle("center domain velocity")
            ax1.set_ylim(0, 1)
            ax1.set_xlim(-0.4, 1.1)
            ax1.grid(visible=True)
            ax1.plot(ydata, y_ind, label="x")
            ax1.legend(loc="upper center")
            ax2.set_ylim(-0.6, 0.45)
            ax2.set_xlim(0, 1)
            ax2.plot(x_ind, xdata, label="y")
            ax2.legend(loc="upper center")
            plt.grid()

            if not store_at == "":
                if not store_at.endswith("/"):
                    store_at += "/"
                plt.savefig(store_at + "post_sim.png")
            if show:
                plt.show()


if __name__ == '__main__':
    sim = LidDrivenCavity()
    sim.run(10000, 2000, "sim_out/LDC")
    sim.serialize("sim_out/LDC/chkpt")
    sim.post_sim(show=True, store_at="sim_out/LDC")
