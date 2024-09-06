from random import randint


class Species:
    """Encapsulates the properties of a chemical species.
       These include its name, the mass of a single particle of this species, its radius and color
       (both for the graphical output).
       """
    inst_count = 0

    def __init__(self, name: str, mass: float, radius: float, color=None):
        self.name = name
        self.mass = mass
        self.radius = radius
        color_int = randint(0, 0xFFFFFF) if color is None else color
        self.color = f"#{color_int:06X}"
        self.id = Species.inst_count
        Species.inst_count += 1
