import random
import math
from typing import Tuple

import pygame
import numpy as np
import matplotlib.pyplot as plt

from Particle import Particle
from Species import Species
from Quadtree import Quadtree

from src.UnitSystem import *


class Buffer:
    def __init__(self, size=5):
        self.size = size
        self.ring = []
        self.cnt = 0

    def insert(self, data):
        if len(self.ring) < self.size:
            self.ring.append(data)
        else:
            self.ring[self.cnt % self.size] = data
        self.cnt += 1

    def all(self, do_dim=False):
        total = []
        for entry in self.ring:
            if not do_dim:
                total += entry
            else:
                total += [mag(dim(x, 'm/s')) for x in entry]
        return total

    def avg(self):
        all = self.all()
        return [s / len(self.ring) for s in all]

class MicroSimulator:
    """Molecular Dynamics simulator for multi-species particle collisions.
       Supports purely elastic (non-reactive) and reactive collisions between two particles.
       To run a simulation, register the required chemical species (register_species), if required the
       reaction rules (register_reaction) and initialize the simulation (init)
    """
    def __init__(self, width: int, height: int, dt: float = 1.0, seed: int = 42, hist_steps: int = 10):
        self.w = int(non_dim(width))
        self.h = int(non_dim(height))
        self.width, self.height = width, height
        self.dt = Q(0.1, 'ps')
        self.step = 0
        self.hist_steps = hist_steps
        self.temp = Q(0, 'K')

        self.spec_reg = {}
        self.reac_reg = []
        self.spec_sim = {}
        self.tree = None

        self.buffer = Buffer(size=25)

        self.bins = {}
        self.axs = []
        self.fig = None
        self.max_speed = 0

        random.seed(seed)
        np.random.seed(seed)

        pygame.init()
        self.screen = pygame.display.set_mode((self.w, self.h))

    def register_species(self, species: "Species") -> None:
        """Register a species for use in simulations"""
        if species.name not in self.spec_reg:
            self.spec_reg[species.name] = species
        else:
            raise ValueError("Species " + species.name + " already registered!")

    def register_reaction(self, r1: str, r2: str, p: str, prob: int) -> None:
        """Register a reaction rule for a particle-particle collision
           Reaction rules consist of two reactant species names, r1 and r2, the product species
           name p and a probability for a collision to be reactive (out of [0, 100])
        """
        self.reac_reg.append((r1, r2, p, prob))

    def init(self, info: dict, temperature: float) -> None:
        """Initialize the simulation according to the info dictionary.
           The keys of this dictionary are the names of previously registered species and its values
           are tuples (n,b) where n is the amount of such particles and b the number of desired bins
           for the speed histograms.
           max_vel is the maximum speed particles of mass 1 can have. Ultimately, this determines the simulation's
           temperature.
        """
        self.spec_sim = info
        # sort by species radius in order to start ran init with the biggest ones
        info_sorted = dict(sorted(info.items(), key=lambda item: self.spec_reg[item[0]].radius, reverse=True))
        self.temperature = temperature
        KB = Q(1.380649E-23, 'J/K')  # Boltzmann constant
        self.tree = Quadtree(Q(0, 'bohr'), Q(0, 'bohr'), self.width, self.height, self.spec_reg[next(iter(info_sorted))].radius)

        for val, key in enumerate(info_sorted):
            self.bins[key] = (info[key][1])
            species = self.spec_reg.get(key)
            radius = species.radius
            i = 0
            speeds = 0
            avg_speed = np.sqrt(2 * temperature * KB / species.mass)
            while i < info[key][0]:
                # add 10% margin to boundaries
                x = np.random.rand() * (self.width - 2.2 * radius) + 1.1 * radius
                y = np.random.rand() * (self.height - 2.2 * radius) + 1.1 * radius
                # Scale max vel given for unit mass to species' particle mass
                v_mag = np.random.rand() * 2 * avg_speed
                v_ang = np.random.rand() * 2 * np.pi
                vx, vy = v_mag * np.cos(v_ang), v_mag * np.sin(v_ang)
                speeds += v_mag

                new_p = Particle(x, y, vx, vy, species)
                if not self.tree.does_collide(new_p):
                    self.tree.add_particle(new_p)
                    i += 1
            print("Average speeds. Actual:", speeds / info[key][0], "Targeted:", avg_speed)
        self.tree.redistribute()

    def __setup_plt(self) -> None:
        """Set up the pyplot window that will show the expected MB distributions and the speed histograms"""
        plt.ion()
        self.fig, self.axs = plt.subplots(nrows=len(self.spec_sim), ncols=1)
        self.axs = np.atleast_1d(self.axs)
        for i, key in enumerate(self.spec_sim):
            species = self.spec_reg[key]
            ax = self.axs[i]
            ax2 = ax.twinx()
            xs, mbs = self.maxwell_boltzmann(species)
            ax2.set_yticks([])
            ax2.set_yticks([], minor=True)
            ax2.set_ylim(0, 0.0007)
            ax2.plot(dim(xs, 'm/s'), mbs, 'k-')
            for r in self.reac_reg:
                if r[2] == key:
                    xs, mbs = self.maxwell_boltzmann(self.spec_reg[r[0]])
                    ax2.plot(dim(xs), mbs, self.spec_reg[r[0]].color)
                    xs, mbs = self.maxwell_boltzmann(self.spec_reg[r[1]])
                    ax2.plot(dim(xs), mbs, self.spec_reg[r[1]].color)
            ax.plot()
        self.fig.tight_layout()

    def maxwell_boltzmann(self, species: "Species") -> Tuple[np.ndarray, np.ndarray]:
        """Computes the expected Maxwell-Boltzmann particle speed distribution of the given species'
           particles currently contained in the simulation (might change due to reactions)
           Returns the speeds and the probability.
           """
        KB = Q(1.380649E-23, 'J/K')  # Boltzmann constant
        energy, cnt = self.tree.particle_energy(species)
        if cnt == 0:
            # If there are no such particles (yet), use running average of temp (minor deviations due to random init)
            temp = self.temp / 2    # TODO: To support more than just a single reaction of form A+B->C replace this
        else:
            temp = energy / (KB * cnt)
            self.temp += temp
        print("MB for species " + species.name + " temp " + str(temp))
        m = species.mass
        speeds = Q(np.linspace(0, 4500, 150), 'm/s')

        coeff = m / (KB * temp)
        distr = coeff * speeds * np.exp(-coeff * speeds ** 2 / 2)
        return speeds, mag(distr)

    def plot_hists(self) -> None:
        """Plots the histograms of the current particle speeds per species"""
        for i, key in enumerate(self.spec_sim):
            species = self.spec_reg[key]
            ax = self.axs[i]
            ax.cla()
            ax.set_xlim(left=0, right=4500)
            ax.set_ylim(0, 0.0007)
            speeds = self.tree.particle_speeds(species)
            speeds = [mag(s) for s in speeds]
            self.buffer.insert(speeds)
            counts, bins = np.histogram(self.buffer.all(), bins=self.bins[key], density=True)
            counts /= len(self.buffer.ring)
            ax.hist(bins[:-1], bins, weights=counts, color=self.spec_reg[key].color, label=species.name, alpha=0.5, stacked=True, density=True)
            ax.set_ylabel("Probability [1]")
            ax.set_xlabel("Velocity [m/s]")
            ax.set_title("Timestep " + str(self.step), y=1.0)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def __sim_step(self) -> None:
        """Perform a simulation step consisting of particle collisions, movement and quadtree redistribution"""
        Quadtree.collide_particles(self.tree, self.width, self.height, self.spec_reg, self.reac_reg, self.dt)
        self.tree.move_particles(self.dt)
        self.tree.redistribute()

    def run(self) -> None:
        """Run the simulation until terminated by user"""
        self.__setup_plt()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if self.step % self.hist_steps == 0:
                self.plot_hists()
            self.__sim_step()
            self.screen.fill((255, 255, 255))
            self.tree.draw_particles(self.screen)
            self.step += 1

            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    set_conversion_mode(ConversionMode.DIM)
    radius, mass, speed, temp = characteristics(Q(1, 'bohr'), Q(1, 'u'), Q(2500, 'm/s'), Q(1, 'K'))

    app = MicroSimulator(200*radius, 200*radius, dt=0.2)
    app.register_species(Species("A", mass, 2*radius, color=0xFF0000))
    app.register_species(Species("B", 2*mass, 3*radius, color=0xFF00))
#    app.register_species(Species("C", 5., 10., color=0xFF))
#    app.register_reaction("A", "B", "C", 20)
    app.init({"A": (200, 30)}, temperature=Q(100, 'K'))
    app.run()
