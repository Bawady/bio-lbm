import numpy as np
import concurrent.futures
from typing import List, Tuple

import pygame


class Quadtree:
    """A quadtree for storing particles of different chemical species.
       Supports collisions (reactive and non-reactive) and the movement of particles
    """
    def __init__(self, x: int, y: int, width: int, height: int, max_radius: float, parent: "Quadtree" = None):
        self.children = []
        self.particles = []
        self.x, self.y = x, y
        self.w, self.h = width, height
        self.parent = parent

        if width >= 2 * max_radius:
            w2 = width / 2
            h2 = height / 2
            self.children.append(Quadtree(x + w2, y, w2, h2, max_radius, parent=self))
            self.children.append(Quadtree(x, y + h2, w2, h2, max_radius, parent=self))
            self.children.append(Quadtree(x + w2, y + h2, w2, h2, max_radius, parent=self))
            self.children.append(Quadtree(x, y, w2, h2, max_radius, parent=self))

    def particle_fits(self, p: "Particle") -> bool:
        """Checks if a particle completely fits into this tree. Returns true if it does, otherwise false"""
        fits_x = (self.x <= p.x - p.species.radius) and (p.x + p.species.radius <= self.x + self.w)
        fits_y = (self.y <= p.y - p.species.radius) and (p.y + p.species.radius <= self.y + self.h)
        return fits_x and fits_y

    def add_particle(self, p: "Particle") -> None:
        """Add the particle p to this tree"""
        fits_child = False
        for c in self.children:
            if c.particle_fits(p):
                c.add_particle(p)
                fits_child = True
        if not fits_child:
            self.particles.append(p)

    def does_collide(self, p: "Particle") -> bool:
        """Checks if particle p collides with any other particle contained in the tree.
           Returns true if that's the case, otherwise false
        """
        for q in self.particles:
            dist = np.sqrt((q.x - p.x) ** 2 + (q.y - p.y) ** 2)
            if dist < (q.species.radius + p.species.radius):
                return True
        for c in self.children:
            if c.does_collide(p):
                return True
        return False

    @staticmethod
    def collide_particles(tree, w: int, h: int, spec_reg: dict, reac_reg: dict, dt: float) -> None:
        """
        Perform collisions of the particles contained in this tree. Performs boundary as well as particle-particle
        collisions (reactive and non-reactive).
        :param w: width of the simulation environment
        :param h: height of the simulation environment
        :param spec_reg: species registry
        :param reac_reg: reaction registry
        :param dt: time step
        """
        i = 0
        while i < len(tree.particles):
            tree.particles[i].boundary_collision(w, h, dt)
            j = i + 1
            while j < len(tree.particles):
                new_particle = tree.particles[i].particle_collision(tree.particles[j], spec_reg, reac_reg, dt=dt)
                if new_particle is not None:
                    # In case a reactive collision occurred
                    tree.particles.append(new_particle)
                    tree.particles.remove(tree.particles[j])
                    tree.particles.remove(tree.particles[i])
                j += 1
            i += 1
        for p in tree.particles:
            for c in tree.children:
                c.collide_with(p, spec_reg, reac_reg, dt)
#        with concurrent.futures.ProcessPoolExecutor() as executor:
#            futures = {executor.submit(Quadtree.collide_particles, c, w, h, spec_reg, reac_reg, dt): c for c in tree.children}
#            for future in concurrent.futures.as_completed(futures):
#                try:
#                    result = future.result()
#                except Exception as e:
#                    print(e)
        for c in tree.children:
            Quadtree.collide_particles(c, w, h, spec_reg, reac_reg, dt)

    def collide_with(self, p: "Particle", spec_reg: dict, reac_reg: dict, dt: float) -> None:
        """Perform the collision of external particle p with particles of this tree.
           Such external particles arise when a particle does not fully fit into a tree,
           but overlaps with its boundary.
        """
        for q in self.particles:
            q.particle_collision(p, spec_reg, reac_reg, dt)
        for c in self.children:
            c.collide_with(p, spec_reg, reac_reg, dt)

    def move_particles(self, dt: float) -> None:
        """Move all particles conatined in this tree by the distance they travel in the time step dt"""
        for p in self.particles:
            p.move(dt)

        for c in self.children:
            c.move_particles(dt)

        for p in self.particles:
            if not self.particle_fits(p) and self.parent is not None:
                self.particles.remove(p)
                self.parent.particles.append(p)

    def redistribute(self) -> None:
        """Redistribute particles contained in this tree such that they are assigned to the correct tree level
           and quadrant. This can become necessary after particles moved.
        """
        for c in self.children:
            for p in self.particles:
                if c.particle_fits(p):
                    self.particles.remove(p)
                    c.particles.append(p)
            c.redistribute()

    def particle_energy(self, species: "Species") -> Tuple[float, int]:
        """Returns the total energy of particles of the given species contained in this tree as well as the
           amount of such particles (allowing to compute the average energy as well)
        """
        energy = 0
        cnt = 0
        for p in self.particles:
            if p.species.id == species.id:
                energy += p.mass() * p.speed() ** 2 / 2
                cnt += 1

        for c in self.children:
            e, c = c.particle_energy(species)
            energy += e
            cnt += c

        return energy, cnt

    def particle_speeds(self, species: "Species") -> List[float]:
        """Returns the speeds of all particles of the given species contained in this tree as list"""
        speeds = []
        for p in self.particles:
            if p.species.id == species.id:
                speeds.append(p.speed())
        for c in self.children:
            s = c.particle_speeds(species)
            speeds += s
        return speeds

    def draw_particles(self, surface: pygame.Surface) -> None:
        """Draw all particles contained in this tree to the given pygame surface"""
        for p in self.particles:
            p.draw(surface)
        for c in self.children:
            c.draw_particles(surface)
