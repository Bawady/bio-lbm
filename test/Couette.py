import numpy as np
import sys

from src.MultiLBM import MultiLBM
import matplotlib.pyplot as plt
from src.UnitSystem import *
from utils.dump_util import *
from utils.geomtry_util import *



class Couette(MultiLBM):
    direction = "y"
    serialize_members = ["reynolds", "conv_cnt"]

    def setup(self) -> None:
        self.dx, self.dt, self.init_density = characteristics(Q(1, 'm'), Q(1, 's'), Q(1, 'kg/m**3'))
        self.viscosity = Q(0.5, 'm**2/s')
        self.conv_cnt = 0

        if self.direction == "x":
            self.x = Q(50, 'm')
            self.y = Q(20, 'm')
            self.solid = np.zeros((int(mag(self.y // self.dx)), int(mag(self.x // self.dx))), dtype=int)
            self.solid[0, :] = np.ones(self.width)
            self.solid[-1, :] = np.ones(self.width)
            self.boundaries = self.boundaries_x
        elif self.direction == "y":
            self.x = Q(20, 'm')
            self.y = Q(50, 'm')
            self.solid = np.zeros((int(mag(self.y // self.dx)), int(mag(self.x // self.dx))), dtype=int)
            self.solid[:, 0] = np.ones(self.height)
            self.solid[:, -1] = np.ones(self.height)
            self.boundaries = self.boundaries_y
        else:
            raise ValueError(self.direction)

        self.add_species("FluidParticle", "fluid", init_quantity=self.init_density, viscosity=self.viscosity)
        self.u_max = self.fluid.lattice.q
        self.reynolds = (self.u_max * self.y) / self.viscosity
        self.fluid.stream = self.fluid.stream_periodic

    def check_convergence(self) -> bool:
        if self.direction == "x":
            y_ind = np.arange(self.height)
            ref = self.u_max * (1 - 1 / (self.height - 1) * y_ind)
            vel = self.fluid.velocity[0, :, self.width // 2]
        else:
            x_ind = np.arange(self.width)
            ref = self.u_max * (1 - 1 / (self.width - 1) * x_ind)
            vel = self.fluid.velocity[1, self.height // 2, :]

        if np.max(np.abs(vel - ref)) <= 5 * sys.float_info.epsilon:
            self.conv_cnt += 1
        else:
            self.conv_cnt = 0

        if self.conv_cnt >= 5:
            return True

    def boundaries_x(self) -> None:
        vel_top = Q(np.zeros((2, self.width), dtype=float), 'm/s')
        vel_top[0, :] = self.u_max
        self.wall_boundary(self.fluid, 0, 0, self.width, self.fluid.lattice.bc_top,
                           velocity_profile=vel_top)
        self.wall_boundary(self.fluid, -1, 0, self.width, self.fluid.lattice.bc_bot,
                           velocity_profile=0)

    def boundaries_y(self) -> None:
        vel_left = Q(np.zeros((2, self.height), dtype=float), 'm/s')
        vel_left[1, :] = self.u_max
        self.wall_boundary(self.fluid, 0, 0, self.height, self.fluid.lattice.bc_left,
                           velocity_profile=vel_left)
        self.wall_boundary(self.fluid, -1, 0, self.height, self.fluid.lattice.bc_right,
                           velocity_profile=0)

    def post_sim(self, show: bool = False, store_at: str = "") -> None:
        dump_as_img(self.fluid.velocity[0 if self.direction == "x" else 1], store_at, f"vel_{self.dump_count}")
        with open(store_at + "/couette.csv", "w") as f:
            if show or not store_at == "":
                if self.direction == "x":
                    y_ind = np.arange(self.height)
                    ref = self.u_max * (1 - 1 / (self.height - 1) * y_ind)
                    data = self.fluid.velocity[0, :, self.width // 2]
                    plt.plot(dim(ref, 'm/s'), y_ind, label="ref")
                    plt.plot(dim(data, 'm/s'), y_ind, label="lbm", linestyle='--', marker='o')
                else:
                    x_ind = np.arange(self.width)
                    ref = self.u_max * (1 - 1 / (self.width - 1) * x_ind)
                    data = self.fluid.velocity[1, self.height // 2, :]
                    plt.plot(x_ind, dim(ref, 'm/s'), label="ref")
                    plt.plot(x_ind, dim(data, 'm/s'), label="lbm", linestyle='--', marker='o')
                plt.legend(loc="upper center")

                if not store_at == "":
                    if not store_at.endswith("/"):
                        store_at += "/"
                    plt.savefig(store_at + "post_sim.png")
                if show:
                    plt.show()

    def dump(self, directory: str) -> None:
        dump_as_img(self.fluid.density, directory, f"density_{self.dump_count}")
        dump_as_img(self.fluid.velocity[0 if self.direction == "x" else 1], directory, f"vel_{self.dump_count}")
        self.dump_count += 1


if __name__ == '__main__':
        sim = Couette()
        sim.run(5000, 1000, "sim_out/Couette")
        sim.post_sim(show=True, store_at="sim_out/Couette")
