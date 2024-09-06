import matplotlib.image as plt_img
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple, TYPE_CHECKING
import pint
from src.UnitSystem import *
import src.Species as Species
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 14})

# For cirular dependence between this and LBM due to type hints
if TYPE_CHECKING:
    from src.MultiLBM import MultiLBM


def get_density(species: Species.Species, params: dict) -> Tuple[str, np.array]:
    name = species.name
    dens = species.density
    return name + "_density", dens


def get_fluid_vel(species: Species.Species, params: dict) -> Tuple[str, np.array]:
    fluid_vel = species.velocity

    if "select" in params:
        select = params["select"]
        if select == "x":
            data = fluid_vel[0]
            suffix = "_x"
        elif select == "y":
            data = fluid_vel[1]
            suffix = "_y"
        elif select == "xy":
            data = np.sqrt(fluid_vel[0] ** 2 + fluid_vel[1] ** 2)
            suffix = "_xy"
        else:
            raise ValueError(select)
    else:
        raise KeyError("select")
    return species.name + "_velocity" + suffix, data


def get_vorticity(species: Species.Species, params: dict) -> Tuple[str, np.array]:
    fluid_vel = species.velocity
    if isinstance(fluid_vel, pint.Quantity):
        fluid_vel = fluid_vel.magnitude
    vorticity = (np.roll(fluid_vel[0], -1, axis=0) - np.roll(fluid_vel[0], 1, axis=0)) - (
        np.roll(fluid_vel[1], -1, axis=1) - np.roll(fluid_vel[1], 1, axis=1))
    return "vorticity", vorticity


def dump_img(species: Species.Species, data_func: Callable, directory: str, file_suffix: str,
             cmap='viridis', mask: np.ndarray = None, **func_params: dict) -> None:
    if directory[-1:] != '/':
        directory += '/'

    type, data = data_func(species, func_params)
    if "unit" in func_params:
        data = dim(data, func_params["unit"])

    if isinstance(data, pint.Quantity):
        vmin = np.min(data.magnitude)
        vmax = np.max(data.magnitude)
    else:
        vmin = np.min(data)
        vmax = np.max(data)

    if mask is not None:
        data *= (1 - mask)

    path = directory + type + "_" + file_suffix + ".png"
    plt_img.imsave(path, data, cmap=cmap, vmin=vmin, vmax=vmax, dpi=600)

def dump_as_img(data: pint.Quantity | np.ndarray, directory: str, file_name: str,
             cmap='viridis', mask: np.ndarray = None) -> None:
    if directory[-1:] != '/':
        directory += '/'

    if isinstance(data, pint.Quantity):
        vmin = np.min(data.magnitude)
        vmax = np.max(data.magnitude)
    else:
        vmin = np.min(data)
        vmax = np.max(data)

    if mask is not None:
        data *= (1 - mask)

    path = directory + file_name + ".png"

    fig, ax = plt.subplots()
    plt.xlabel("x [1]", loc="center")
    plt.ylabel("y [1]", loc="center")
    cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
#    divider = make_axes_locatable(ax)
#    cbar_ax = divider.append_axes("bottom", size="5%", pad=0.05)
#    cbar = fig.colorbar(cax, cax=cbar_ax)
#    cbar_ax.set_aspect(20)
#    cbar.set_label("u_x / v_wall [1]", loc="center")

    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)

#    plt.colorbar()
#    plt_img.imsave(path, data, cmap=cmap, vmin=vmin, vmax=vmax, dpi=600)
