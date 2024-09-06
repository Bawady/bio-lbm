from src.MultiLBM import MultiLBM
from src.UnitSystem import *
import matplotlib.pyplot as plt
from utils.dump_util import *
from utils.geomtry_util import *
import src.Lattices as Lattices
import src.Species as Species


class SubLattice(MultiLBM):

    serialize_members = ["k"]

    def setup(self) -> None:
        self.k = 2
        self.dxa, self.dt, self.initial_density = characteristics(Q(1, 'm'), Q(1, 's'), Q(1, 'kg/m**3'))
        self.dxb = self.k * self.dxa
        viscosity = Q(1 / 6., 'm**2/s')

        self.x = Q(500, 'm')
        self.y = Q(500, 'm')

#        self.lattices["lta"] = Lattices.factory.create("D2Q9", dx=self.dxa, dt=self.dt)
        self.lattices["ltb"] = Lattices.factory.create("D2Q9", dx=self.dxb, dt=self.dt)

#        self.widtha, self.heighta = int(mag(self.x / self.dxa)), int(mag(self.y / self.dxa))
        self.widthb, self.heightb = int(mag(self.x / self.dxb)), int(mag(self.y / self.dxb))

#        self.species_registry["speca"] = Species.factory.create("BasicParticle", (self.heighta, self.widtha), self.lattices["lta"], #{"viscosity": viscosity, "init_quantity": self.initial_density})
#        self.speca = self.species_registry["speca"]
        self.species_registry["specb"] = Species.factory.create("BasicParticle", (self.heightb, self.widthb), self.lattices["ltb"], {"viscosity": 2*viscosity, "init_quantity": self.initial_density})
        self.specb = self.species_registry["specb"]

#        self.speca.density[:, 0:self.widtha//2] *= 2
        self.specb.density[:, self.widthb//2:] *= 2
        self.bulk_velocity = self.specb.velocity

    def boundaries(self) -> None:
#        self.wall_boundary(self.speca, 0, 0, self.heighta, self.speca.lattice.bc_left,
#                           velocity_profile=Q(0, 'm/s'))
#        self.wall_boundary(self.speca, -1, 0, self.heighta, self.speca.lattice.bc_right,
#                           velocity_profile=Q(0, 'm/s'))
#        self.wall_boundary(self.speca, 0, 0, self.widtha, self.speca.lattice.bc_top,
#                           velocity_profile=Q(0, 'm/s'))
#        self.wall_boundary(self.speca, -1, 0, self.widtha, self.speca.lattice.bc_bot,
#                           velocity_profile=Q(0, 'm/s'))

        self.wall_boundary(self.specb, 0, 0, self.heightb, self.specb.lattice.bc_left,
                           velocity_profile=Q(0, 'm/s'))
        self.wall_boundary(self.specb, -1, 0, self.heightb, self.specb.lattice.bc_right,
                           velocity_profile=Q(0, 'm/s'))
        self.wall_boundary(self.specb, 0, 0, self.widthb, self.specb.lattice.bc_top,
                           velocity_profile=Q(0, 'm/s'))
        self.wall_boundary(self.specb, -1, 0, self.widthb, self.specb.lattice.bc_bot,
                           velocity_profile=Q(0, 'm/s'))

    def dump(self, directory: str) -> None:
        fig, ax1 = plt.subplots(1, 1)
        fig.suptitle("center domain velocity")
        ax1.set_ylim([-0.2, 2.2])
        ax1.plot(self.specb.density[self.heightb//2, :], 'r', label="A")
        ax1.legend(loc="upper center")
#        ax2.plot(self.specb.density[self.heightb//2, :], 'b', label="B")
#        ax2.legend(loc="upper center")

        if not directory.endswith("/"):
            directory += "/"
        plt.savefig(directory + "dump_" + str(self.dump_count) + ".png")
        plt.close()
        self.dump_count += 1

if __name__ == '__main__':
    sim = SubLattice()
    sim.run(50, 2, "sim_out/SubLattice/new")
    sim.post_sim(show=True, store_at="sim_out/SubLattice/new")
