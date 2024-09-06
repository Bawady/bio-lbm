import numpy as np
import os
from functools import wraps

import src.Lattices as Lattices
import src.Species as Species
from src.Factory import Fabricable
from src.Serializable import Serializable
from utils.dump_util import *
from utils.geomtry_util import *
from src.UnitSystem import *
from src.Reactions.CRN import CRN

# For circular dependence due to type hints
if TYPE_CHECKING:
    from .MultiLBM import MultiLBM


def delegate_to(member):
    def decorator(func):
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            obj = getattr(instance, member)
            method = getattr(obj, func.__name__)
            return method(*args, **kwargs)
        return wrapper
    return decorator


class MultiLBM(Fabricable, Serializable):
    serialize_members = ["x", "y", "dx", "dt", "width", "height", "lattices", "species_registry", "solid", "bulk_density",
                         "bulk_velocity", "runs", "dump_count", "init_done", "react_params"]

    def __init__(self):
        self.dx, self.dt = None, None
        self.width, self.height = None, None
        self.lattices = {}
        self.species_registry = {}
        self.reaction_system = None
        self.solid = None
        self.bulk_density, self.bulk_velocity = None, None
        self.runs = 0
        self.dump_count = 0
        self.init_done = False
        self.react_params = {}

    def __pre_setup(self):
        self.reaction_system = CRN()
        self.detector = WallDetector()

    def setup(self) -> None:
        pass

    def __post_setup(self):
        width, height = int(mag(self.x / self.dx)), int(mag(self.y / self.dx))
        self.width = int(non_dim(width) if isinstance(width, pint.Quantity) else width)
        self.height = int(non_dim(height) if isinstance(height, pint.Quantity) else height)

        if self.solid is None:
            self.solid = np.zeros((self.height, self.width), dtype=int)
        self.detector.detect(self.solid)

    def init_state(self):
        pass

    def sim_step(self) -> None:
        self.update_moments()
        self.update_feq()
        self.collide()
        self.reaction_system.react(**self.react_params)
        self.stream()
        self.boundaries()
        self.runs += 1

    def time(self, unit: str = 's'):
        return dim(Q(self.runs, '1') * self.dt, unit)

    def init(self):
        self.__pre_setup()
        self.setup()
        self.__post_setup()
        self.init_state()
        self.__init_pops()
        self.init_done = True

    def check_convergence(self) -> bool:
        return False

    def run(self, runs: int, dump_period: int, out_dir: str) -> None:
        if not self.init_done:
            self.init()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        dumps = runs // dump_period
        i = 0
        while i < runs and not self.check_convergence():
            if i % dump_period == 0:
                self.dump(out_dir)
                print(f"{i//dump_period + 1} / {dumps}")
            self.sim_step()
            i += 1

    def init_sim_params(self, params: dict) -> None:
        pass

    def __init_non_primitive(self) -> None:
        pass

    def calculate_sim_params(self) -> None:
        pass

    def init_geometry(self) -> None:
        pass

    def add_lattice(self, lattice: str) -> None:
        if lattice not in self.lattices:
            self.lattices[lattice] = Lattices.factory.create(lattice, dx=self.dx, dt=self.dt)

    def get_lattice(self, name: str) -> Lattices.Lattice:
        if name not in self.lattices:
            raise KeyError(f"No lattice{name} was added to simulation")
        return self.lattices[name]

    @delegate_to("reaction_system")
    def add_reaction(self, reactants: dict, products: dict, kinetics: Callable, **params: dict):
        pass

    def add_species(self, species: str | Callable, name: str, lattice: str = "D2Q9", **params: dict) -> Species.Species:
        if name in self.species_registry:
            return self.species_registry[name]

        if name not in params:
            params["name"] = name

        if lattice not in self.lattices:
            self.add_lattice(lattice)
        size = (int(mag(self.y / self.dx)), int(mag(self.x / self.dx)))
        if isinstance(species, Callable):
            spec = species()
            spec.init(size, self.lattices[lattice], params)
        else:
            spec = Species.factory.create(species, size, self.lattices[lattice], params=params)
        self.species_registry[name] = spec
        self.__setattr__(name, spec)
        return spec

    def get_species(self, name: str) -> Species.Species:
        if name not in self.species_registry:
            raise KeyError(f"No species {name} was added to simulation")
        return self.species_registry[name]

    def __init_pops(self) -> None:
        for s in self.species_registry.values():
            s.init_populations(self.bulk_velocity)

    def update_bulk_moments(self) -> None:
        pass

    def update_moments(self) -> None:
        for s in self.species_registry.values():
            s.update_moments()
        self.update_bulk_moments()

    def update_feq(self) -> None:
        for s in self.species_registry.values():
            s.update_feq(self.bulk_velocity)

    def stream_periodic(self):
        for s in self.species_registry.values():
            s.stream_periodic()

    def collide(self) -> None:
        for s in self.species_registry.values():
            s.collide_self()

    def stream(self) -> None:
        for s in self.species_registry.values():
            s.stream()

    def boundaries(self) -> None:
        pass

    # Allows comparison of LBM with reference (e.g. analytical solution) or final output
    def post_sim(self, show: bool, store_at: str = "") -> None:
        pass

    # Dump simulation data
    def dump(self, directory: str) -> None:
        pass

    def corner_boundary(self, species: Species.Species, y: int, x: int, boundary_function: Callable) -> None:
        vel = self.bulk_velocity if self.bulk_velocity is not None else species.velocity
        boundary_function(species.f, species.density, vel, y, x)

    def wall_boundary(self, species: Species.Species, sel: int, start: int, end: int, boundary_function: Callable,
                      velocity_profile: np.ndarray = None, density_profile: np.ndarray = None) -> None:
        vp = velocity_profile
        dp = density_profile
        vel = self.bulk_velocity if self.bulk_velocity is not None else species.velocity
        if velocity_profile is None and density_profile is None:
            vp = np.zeros((2, end-start))
        boundary_function(species.f, species.density, vel,
                          sel, start, end, velocity_profile=vp, density_profile=dp)

    def __corner_boundaries(self, species: Species, corners: Tuple, boundary_function: Callable) -> None:
        vel = self.bulk_velocity if self.bulk_velocity is not None else species.velocity
        for i in range(len(corners[0])):
            boundary_function(species.f, species.density, vel, corners[0][i], corners[1][i])

    def __wall_boundaries(self, species: Species, segments: np.ndarray, boundary_function: Callable) -> None:
        vel = self.bulk_velocity if self.bulk_velocity is not None else species.velocity
        for s in segments:
            a, b, c = s
            c += 1
            boundary_function(species.f, species.density, vel,
                              a, b, c, velocity_profile=Q(np.zeros((2, c-b)), 'm/s'))

    def __solid_boundaries(self) -> None:
        for species in self.species_registry.values():
            lt = species.lattice

            self.__wall_boundaries(species, self.detector.top_walls, lt.bc_top)
            self.__wall_boundaries(species, self.detector.bot_walls, lt.bc_bot)
            self.__wall_boundaries(species, self.detector.right_walls, lt.bc_right)
            self.__wall_boundaries(species, self.detector.left_walls, lt.bc_left)

            self.__corner_boundaries(species, self.detector.conc_bl, lt.bc_concave_bot_left)
            self.__corner_boundaries(species, self.detector.conc_tl, lt.bc_concave_top_left)
            self.__corner_boundaries(species, self.detector.conc_tr, lt.bc_concave_top_right)
            self.__corner_boundaries(species, self.detector.conc_br, lt.bc_concave_bot_right)

            self.__corner_boundaries(species, self.detector.conv_bl, lt.bc_convex_bot_left)
            self.__corner_boundaries(species, self.detector.conv_tl, lt.bc_convex_top_left)
            self.__corner_boundaries(species, self.detector.conv_tr, lt.bc_convex_top_right)
            self.__corner_boundaries(species, self.detector.conv_br, lt.bc_convex_bot_right)

    @classmethod
    def deserialize(cls, chkpt_dir: str, obj_dict: dict = None, chkpt_subdir: str = "") -> typing.Any:
        if chkpt_dir[-1:] != '/':
            chkpt_dir += '/'

        type, id, chkpt = Serializable.load_checkpoint(chkpt_dir + chkpt_subdir)
        mlbm = MultiLBM.factory.create_blank(type)
        mlbm.__pre_setup()
        mlbm.load_members(chkpt_dir, chkpt, id, obj_dict, chkpt_subdir)
        mlbm.setup()
        mlbm.__post_setup()
        return mlbm
