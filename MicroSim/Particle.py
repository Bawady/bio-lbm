import numpy as np
import random
from typing import Tuple

import numpy as np
import pygame

from Species import Species
from src.UnitSystem import *


class Particle:
    """
    Encapsulates the behavior and properties of a single particle.
    A particle essentially consists of a position, velocity and chemical species.
    Includes collision (non-)reactive collision handling.
    """

    id_count = 0

    def __init__(self, x: float, y: float, vx: float, vy: float, species: "Species"):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.species = species
        self.id = Particle.id_count
        Particle.id_count += 1

    def move(self, dt: float = 1) -> None:
        """Move this particle by the distance it travels in time dt"""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def boundary_collision(self, width: float, height: float, dt: float = 1) -> None:
        """Checks if the particle would leave the simulation environment of given width and height
            within the next timestep of dt. If it does, it bounces of the wall.
        """
        radius = self.species.radius
        xmin = self.x - radius
        xmax = self.x + radius
        ymin = self.y - radius
        ymax = self.y + radius
        if (xmax + self.vx * dt) >= width or (xmin + self.vx * dt) <= Q(0, 'bohr'):
            self.vx = -self.vx
        if (ymax + self.vy * dt) >= height or (ymin + self.vy * dt) <= Q(0, 'bohr'):
            self.vy = -self.vy

    def particle_collision(self, p, spec_reg, reac_reg, dt: float = 1) -> "Particle":
        """Checks if this particle and p collide and if they do perform the collision.
           Returns None in case of a non-reactive collision or no collision and in case of
           a reactive collision the new particle
        """
        v1 = Q(np.array([mag(self.vx), mag(self.vy)]), unit(self.vx))
        v2 = Q(np.array([mag(p.vx), mag(p.vy)]), unit(p.vx))
        m1, m2 = self.species.mass, p.species.mass
        di = Q(np.array([mag(self.x - p.x), mag(self.y - p.y)]), unit(self.x))
        norm = np.linalg.norm(di)
        x1_next = Q(np.array([mag(self.x), mag(self.y)]), unit(self.x)) + dt * v1
        x2_next = Q(np.array([mag(p.x), mag(p.y)]), unit(p.x)) + dt * v2
        norm_next = np.linalg.norm(x1_next - x2_next)

        if self.species.radius + p.species.radius > norm > norm_next:
            ran = random.randrange(0, 100)
            for r in reac_reg:
                if ((r[0] == self.species.name and r[1] == p.species.name) or
                    (r[1] == self.species.name and r[0] == p.species.name)) and ran < r[3]:
                    return self.__reaction_product(p, r, spec_reg)

            u1 = v1 - 2. * m2 / (m1 + m2) * np.dot(v1 - v2, di) / (np.linalg.norm(di) ** 2.) * di
            u2 = v2 + 2. * m1 / (m2 + m1) * np.dot(v2 - v1, (-di)) / (np.linalg.norm(di) ** 2.) * di
            self.vx, self.vy = u1
            p.vx, p.vy = u2
        return None

    def __reaction_product(self, p: "Particle", reaction: Tuple[str, str, str, int], spec_reg: dict) -> "Particle":
        """Determine the product particle of a reactive collision for the reaction
           A+B->C as [A, B, C, probability * 100]
        """
        product_spec = spec_reg[reaction[2]]
        if self.species.radius > p.species.radius:
            bigger = self
        else:
            bigger = p
        new_v = np.sqrt(2 * (self.kinetic_energy() + p.kinetic_energy()) / product_spec.mass)
        theta = random.randint(0, 200) / 100.0 * np.pi
        new_vx = new_v * np.cos(theta)
        new_vy = new_v * np.sin(theta)
        return Particle(bigger.x, bigger.y, new_vx, new_vy, product_spec)

    def speed(self) -> float:
        """Return the magnitude of the particle velocity"""
        return np.sqrt(self.vx ** 2 + self.vy ** 2)

    def mass(self) -> float:
        """Return the mass of this particle"""
        return self.species.mass

    def draw(self, surface: pygame.Surface):
        """Draw this particle to the pygame surface"""
        species = self.species
        pygame.draw.circle(surface, species.color, (int(non_dim(self.x)), int(non_dim(self.y))), int(non_dim(species.radius)))

    def kinetic_energy(self) -> float:
        """Return the kinetic energy of this particle in units [mass] * [speed]**2"""
        return self.mass() * self.speed() ** 2 / 2.0
