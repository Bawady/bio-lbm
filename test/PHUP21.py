from src.MultiLBM import MultiLBM
from src.UnitSystem import *
from utils.dump_util import *
from utils.geomtry_util import *
import matplotlib.pyplot as plt


class PHUP21(MultiLBM):
    def setup(self) -> None:
        self.x, self.dt, self.density = characteristics(Q(100, 'm'), Q(1, 's'), Q(1, 'kg/m**3'))
        self.reynolds = 200
        self.peclet = 150
        self.init_density = Q(1, 'kg/m**3')
        nodes = 150
        self.dx = self.x / nodes
        self.y = Q(100, 'm')
        self.u_max = Q(0.1, 'm/s')
        viscosity = self.u_max * self.x / self.reynolds
        diffusivity = self.u_max * self.x / self.peclet
        self.sigma = math.floor(10 * self.x)

        fluid = self.add_species("FluidParticle", "fluid", "D2Q9", init_quantity=self.init_density, viscosity=viscosity)
        speca = self.add_species("RDEParticle", "speca", "D2Q9", init_quantity=Q(0, 'kg/m**3'), viscosity=diffusivity)
        specb = self.add_species("RDEParticle", "specb", "D2Q9", init_quantity=Q(0, 'kg/m**3'), viscosity=diffusivity)
        specc = self.add_species("RDEParticle", "specc", "D2Q9", init_quantity=Q(0, 'kg/m**3'), viscosity=diffusivity)
        self.bulk_velocity = self.fluid.velocity
        self.add_reaction([1 * speca, 1 * specb], [1 * specc], self.reaction_rate)

    @staticmethod
    def reaction_rate(reactants, externals, products):
        return Q(0.01, 'm**3/(kg * s)')

    def init_geometry(self) -> None:
        solid_rectangle(0, self.width, 0, self.height, self.solid)

    def wall_velocity(self, size: int, dir: int) -> np.ndarray:
        velocity = Q(np.zeros((2, size)), 'm/s')
        velocity[dir] = self.u_max * (1 - math.exp(-self.runs**2 / (2 * self.sigma**2)))
        return velocity

    def boundaries(self) -> None:
        for species in [self.fluid, self.speca, self.specb, self.specc]:
            self.wall_boundary(species, -1, 0, self.width,
                               species.lattice.bc_bot, velocity_profile=0)
            self.wall_boundary(species, 0, 0, self.height,
                               species.lattice.bc_left, velocity_profile=0)
            self.wall_boundary(species, -1, 0, self.height,
                               species.lattice.bc_right, velocity_profile=0)
            if species == self.fluid:
                self.wall_boundary(species, 0, 0, self.width,
                                   species.lattice.bc_top, velocity_profile=self.wall_velocity(self.width, 0))
                self.corner_boundary(species, -1, 0, species.lattice.bc_concave_bot_left)
                self.corner_boundary(species, 0, 0, species.lattice.bc_concave_top_left)
                self.corner_boundary(species, 0, -1, species.lattice.bc_concave_top_right)
                self.corner_boundary(species, -1, -1, species.lattice.bc_concave_bot_right)
            else:
                self.wall_boundary(species, 0, 0, self.width,
                                   species.lattice.bc_top, velocity_profile=0)

    def init_state(self) -> None:
        filled_rectangle(0, self.width//2, 0, self.height, self.speca.density, self.density)
        filled_rectangle( self.width//2, self.width, 0, self.height, self.specb.density, self.density)

    def dump(self, directory: str) -> None:
        dump_as_img(self.speca.density, directory, "densA_" + str(self.dump_count), cmap="Blues")
        dump_as_img(self.specb.density, directory, "densB_" + str(self.dump_count), cmap="Reds")
        dump_as_img(self.specc.density, directory, "densC_" + str(self.dump_count), cmap="Greens")
        dump_as_img(np.sqrt(self.fluid.velocity[0]**2 + self.fluid.velocity[1]**2), directory, "fvel" + str(self.dump_count), cmap="jet")
        self.dump_count += 1


if __name__ == '__main__':
    sim = PHUP21()
    sim.run(10000, 1000, "sim_out/PHUP21")
    sim.serialize("sim_out/PHUP21")