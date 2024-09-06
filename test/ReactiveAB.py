from src.MultiLBM import MultiLBM
from src.UnitSystem import *
from utils.dump_util import *
from utils.geomtry_util import *
import matplotlib.pyplot as plt


# Reproduce paper by Qian and Orszag about simulation of RDE A+B->C
# Collision: BGK-based with additional term
# Equilibrium: Uniform
class ReactiveAB(MultiLBM):
    def init_sim_params(self, params: dict) -> None:
        self.x = Quantity(200, 'm')
        self.y = Quantity(200, 'm')
        self.pops = 2
        self.dx = Quantity(1, 'um')
        self.dt = Quantity(1, 'ms')
        self.viscosity = Quantity(0.129, "m**2/s")
        self.reaction_rate = Quantity(0.001, '') # TODO: Plug in correct unit
        self.density_a = Quantity(1., 'kg/m**3')
        self.density_b = Quantity(1., 'kg/m**3')
        self.u_max = Quantity(0.6, 'm/s')

    def init_geometry(self) -> None:
        solid_rectangle(0, self.width, 0, self.height, self.solid)
        self.top_wall   = Quantity(np.zeros((2, self.width)), 'm/s')
        self.bot_wall   = Quantity(np.zeros((2, self.width)), 'm/s')
        self.left_wall  = Quantity(np.zeros((2, self.height)), 'm/s')
        self.right_wall = Quantity(np.zeros((2, self.height)), 'm/s')

    def collide(self) -> None:
        # Non-reacting BGK part
        super().collide()
        reactive_term = self.reaction_rate * self.density[0] * self.density[1] / self.lattice.Q
        self.f_coll[0] += reactive_term
        self.f_coll[1] -= reactive_term

    def init_state(self) -> None:
        self.density = np.zeros((self.pops+1, self.height, self.width), dtype=float) * self.initial_density
        solid_circle_filled(self.width // 2, self.height // 2, self.width // 8, self.density[0], self.density_a)
        filled_rectangle(0, self.width, 0, self.height, self.density[1], self.density_b)

        self.fluid_vel = Quantity(np.zeros((self.pops+1, 2, self.height, self.width), dtype=float), "m/s")
        self.feq = Quantity(np.zeros((self.pops, self.lattice.Q, self.height, self.width), dtype=float), "kg/m**3")
        for p in range(self.pops):
            self.update_feq(p)
        self.f = self.feq

    def update_moments(self) -> None:
        q = self.lattice.q
        for p in range(self.pops):
            f = self.f[p]
            self.density[p] = np.sum(f, axis=0)
            self.fluid_vel[p, 0] = q * np.sum(self.lattice.dir_x[:, None, None] * f, axis=0)
            self.fluid_vel[p, 1] = q * np.sum(self.lattice.dir_y[:, None, None] * f, axis=0)
        self.density[-1] = np.sum(self.density[0:self.pops], axis=0)
        self.fluid_vel[-1] = np.sum(self.fluid_vel[0:self.pops], axis=0)
        self.fluid_vel /= self.density[-1]

    def boundaries(self) -> None:
        vel_profile = Quantity(np.zeros((2, self.width)), 'm/s')
        vel_profile[0, :] = self.u_max * Quantity((1 - math.exp(-self.runs**2 / (2 * 1E04))), '')
        for p in range(self.pops):
            self.default_boundaries(p)
            self.wall_boundary(p, 0, 0, self.width, self.lattice.bc_top, velocity_profile=vel_profile)

    def dump(self, directory: str) -> None:
        for p in range(self.pops):
            dump_img(self, directory, get_density, prefix=chr(ord("A") + p), p=p, cmap='RdBu')

        path = directory + "/species" + str(self.dump_count) + ".png"
        max = np.max(self.density)
        rgb_data = np.zeros((self.height, self.width, 3), dtype=float)
        w, h, c = rgb_data.shape
        for iy in range(h):
            for ix in range(w):
                rgb_data[iy, ix, 0] = self.density[0, iy, ix] / max
                rgb_data[iy, ix, 1] = self.density[1, iy, ix] / max
                rgb_data[iy, ix, 2] = 0.0
        plt_img.imsave(path, rgb_data, dpi=600)

        self.dump_count += 1

    def post_sim(self, show: bool, store_at: str = "") -> None:
        if show or not store_at == "":
            max = np.max(self.density[-1])
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle("center domain velocity")
            ax1.set_ylim([0, max])
            ax1.plot(self.density[0, self.height//2, :], 'r', label="A")
            ax1.legend(loc="upper center")
            ax2.set_ylim([0, max])
            ax2.plot(self.density[1, self.height//2, :], 'b', label="B")
            ax2.legend(loc="upper center")
            ax3.set_ylim([0, max])
            ax3.plot(self.density[2, self.height//2, :], 'g', label="C")
            ax3.legend(loc="upper center")

            if not store_at == "":
                if not store_at.endswith("/"):
                    store_at += "/"
                plt.savefig(store_at + "post_sim.png")
            if show:
                plt.show()
