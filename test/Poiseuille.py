from src.MultiLBM import MultiLBM
from src.UnitSystem import *
import matplotlib.pyplot as plt
from utils.dump_util import *
from utils.geomtry_util import *
import test
import sys


class Poiseuille(MultiLBM):
    serialize_members = ["direction",  "reynolds"]

    direction = "y"

    def setup(self) -> None:
        self.dx, self.dt, self.init_density = characteristics(Q(1, 'm'), Q(1, 's'), Q(1, 'kg/m**3'))
        self.reynolds = 100
        self.x = 80 * self.dx
        self.y = 20 * self.dx
        self.u_lbm = Q(0.2, 'm/s')
        self.u_avg = 2/3 * self.u_lbm
        viscosity = self.u_avg * self.y / self.reynolds
        self.sigma = math.floor(10 * self.x)

        self.add_species("FluidParticle", "fluid", "D2Q9", viscosity=viscosity,
                         init_quantity=self.init_density)
        print(self.fluid.tau)

    def boundaries(self) -> None:
        if self.direction == "x":
            self.wall_boundary(self.fluid, -1, 0, self.width, self.fluid.lattice.bc_bot,
                               velocity_profile=Q(0, 'm/s'))
            self.wall_boundary(self.fluid, 0, 0, self.height, self.fluid.lattice.bc_left,
                              velocity_profile=self.inlet_velocity(self.height))
            self.wall_boundary(self.fluid, 0, 0, self.width, self.fluid.lattice.bc_top,
                               velocity_profile=Q(0, 'm/s'))
            self.wall_boundary(self.fluid, -1, 0, self.height, self.fluid.lattice.bc_right,
                               density_profile=self.init_density)
        elif self.direction == "y":
            self.wall_boundary(self.fluid, -1, 0, self.width, self.fluid.lattice.bc_bot,
                               density_profile=self.init_density)
            self.wall_boundary(self.fluid, 0, 0, self.height, self.fluid.lattice.bc_left,
                               velocity_profile=Q(0, 'm/s'))
            self.wall_boundary(self.fluid, 0, 0, self.width, self.fluid.lattice.bc_top,
                               velocity_profile=self.inlet_velocity(self.width))
            self.wall_boundary(self.fluid, -1, 0, self.height, self.fluid.lattice.bc_right,
                               velocity_profile=Q(0, 'm/s'))
        self.corner_boundary(self.fluid, -1, 0, self.fluid.lattice.bc_concave_bot_left)
        self.corner_boundary(self.fluid, 0, 0, self.fluid.lattice.bc_concave_top_left)
        self.corner_boundary(self.fluid, 0, -1, self.fluid.lattice.bc_concave_top_right)
        self.corner_boundary(self.fluid, -1, -1, self.fluid.lattice.bc_concave_bot_right)

    def check_convergence(self) -> bool:
        if self.direction == "x":
            y_ind = np.arange(self.height)
            ref = 4 / (self.height-1) ** 2 * y_ind * (self.height - 1 - y_ind)
            vel = self.fluid.velocity[0, :, self.width//2] / self.u_lbm
        else:
            x_ind = np.arange(self.width)
            ref = 4 / (self.width - 1) ** 2 * x_ind * (self.width - 1 - x_ind)
            vel = self.fluid.velocity[1, self.height // 2, :] / self.u_lbm

        if np.max(np.abs(vel - ref)) <= 0.01:
            self.conv_cnt += 1
        else:
            self.conv_cnt = 0

        if self.conv_cnt >= 10:
            return True

    def inlet_velocity(self, size: int) -> np.ndarray:
        inlet_velocity = Q(np.zeros((2, size)), 'm/s')
        ind = np.arange(size)
        scale = (1.0 - math.exp(-self.runs**2/(2.0*self.sigma**2)))
        inlet_velocity[0 if self.direction == "x" else 1] = 4 * scale * self.u_lbm / (size-1)**2 * ind * (size - 1 - ind)
        return inlet_velocity

    def post_sim(self, show: bool = False, store_at: str = "") -> None:
        with open(store_at + "/couette.csv", "w") as f:
            if show or not store_at == "":
                if self.direction == "x":
                    y_ind = np.arange(self.height)
                    ref = 4 / (self.height-1) ** 2 * y_ind * (self.height - 1 - y_ind)
                    data = self.fluid.velocity[0, :, self.width//2] / self.u_lbm
                    scaled = data * np.max(ref) / np.max(data)
                    plt.plot(ref, y_ind, label="ref")
                    plt.plot(scaled, y_ind, label="lbm", linestyle='--', marker='o')
                else:
                    x_ind = np.arange(self.width)
                    ref = 4 / (self.width - 1) ** 2 * x_ind * (self.width - 1 - x_ind)
                    data = self.fluid.velocity[1, self.height // 2, :] / self.u_lbm
                    plt.plot(x_ind, ref, label="ref")
                    plt.plot(x_ind, data, label="lbm", linestyle='--', marker='o')
                plt.legend(loc="upper right")

                if not store_at == "":
                    if not store_at.endswith("/"):
                        store_at += "/"
                    plt.savefig(store_at + "post_sim.png")
                if show:
                    plt.show()

    def dump(self, directory: str) -> None:
        cnt = str(self.dump_count)
        y_ind = np.arange(self.height)
        fig, ax = plt.subplots()
        plt.xlabel("x [1]", loc="center")
        plt.ylabel("y [1]", loc="center")
        ax.set_xlim(0, 3.4)
        ref = 4 / (self.height-1) ** 2 * y_ind * (self.height - 1 - y_ind)
        data = self.fluid.velocity[0, :, self.width//2]
        ax.plot(ref, y_ind, label="ref")
        ax.plot(data, y_ind, label="lbm")
        if not directory.endswith("/"):
            directory += "/"

        fig.savefig(directory + "post_sim" + cnt + ".png", dpi=600, bbox_inches='tight')
        plt.close(fig)

        self.dump_count += 1

if __name__ == '__main__':
    sim = Poiseuille()
    sim.run(5000, 500, "sim_out/Poiseuille")
    sim.post_sim(show=True, store_at="sim_out/Poiseuille")