import math
import numpy as np
import pint
import matplotlib.image as plt_img
from typing import List
from PIL import Image

from src.Serializable import Serializable
from src.UnitSystem import *


def solid_rectangle(x_start, x_end, y_start, y_end, solid, val=1):
    solid[y_start:y_end, x_start] = val
    solid[y_start:y_end, x_end - 1] = val
    solid[y_start, x_start:x_end] = val
    solid[y_end - 1, x_start:x_end] = val

def filled_rectangle(x_start, x_end, y_start, y_end, solid, val=1):
    solid[y_start:y_end, x_start:x_end] = val

def solid_circle(x, y, r, solid, val=1):
    for i in np.arange(0, 350.0, .1):
        x_off = int(r * math.cos(i * math.pi / 180))
        y_off = int(r * math.sin(i * math.pi / 180))
        if is_in_bounds(x + x_off, y + y_off, solid):
            solid[y + y_off, x + x_off] = val


def is_in_bounds(x: int, y: int, solid: np.ndarray) -> bool:
    if isinstance(solid, pint.Quantity):
        (height, width) = solid.magnitude.shape
    else:
        (height, width) = solid.shape
    return (width-1 > x > 0) and (height-1 > y > 0)


def solid_circle_filled(x, y, r, solid, val=1):
    for y1 in range(y - r, y + r, 1):
        for x1 in range(x - r, x + r, 1):
            if is_in_bounds(x1, y1, solid) and ((x1 - x) ** 2 + (y1 - y) ** 2) < r ** 2:
                solid[y1, x1] = val


def img_to_solid(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    solid = np.where(img_array == 0, 1, 0)
    return solid


class WallDetector:
    def __init__(self):
        pass

    def detect(self, solid: np.ndarray) -> None:
        if isinstance(solid, pint.Quantity):
            solid = solid.magnitude

        self._height, self._width = solid.shape

        _conc_bl = self.convolve(solid, self._kernel_conc_bl)
        _conc_br = self.convolve(solid, self._kernel_conc_br)
        _conc_tl = self.convolve(solid, self._kernel_conc_tl)
        _conc_tr = self.convolve(solid, self._kernel_conc_tr)
        _conv_bl = self.convolve(solid, self._kernel_conv_bl)
        _conv_br = self.convolve(solid, self._kernel_conv_br)
        _conv_tl = self.convolve(solid, self._kernel_conv_tl)
        _conv_tr = self.convolve(solid, self._kernel_conv_tr)
        _b = self.convolve(solid, self._kernel_bot)
        _t = self.convolve(solid, self._kernel_top)
        _r = self.convolve(solid, self._kernel_right)
        _l = self.convolve(solid, self._kernel_left)

        self.conc_bl = np.vstack(np.where(_conc_bl))
        self.conc_br = np.vstack(np.where(_conc_br))
        self.conc_tl = np.vstack(np.where(_conc_tl))
        self.conc_tr = np.vstack(np.where(_conc_tr))
        self.conv_bl = np.vstack(np.where(_conv_bl))
        self.conv_br = np.vstack(np.where(_conv_br))
        self.conv_tl = np.vstack(np.where(_conv_tl))
        self.conv_tr = np.vstack(np.where(_conv_tr))

        nconv = (1 - (_conv_br + _conv_tl + _conv_bl + _conv_tr))

        _l += np.roll(_conc_bl, -1, axis=0) + np.roll(_conc_tl, 1, axis=0)
        _l *= nconv
        self.wall_l = np.vstack(np.where(_l))

        _r += np.roll(_conc_br, -1, axis=0) + np.roll(_conc_tr, 1, axis=0)
        _r *= nconv
        self.wall_r = np.vstack(np.where(_r))

        _t += np.roll(_conc_tl, 1, axis=1) + np.roll(_conc_tr, -1, axis=1)
        _t *= nconv
        self.wall_t = np.vstack(np.where(_t))

        _b += np.roll(_conc_bl, 1, axis=1) + np.roll(_conc_br, -1, axis=1)
        _b *= nconv
        self.wall_b = np.vstack(np.where(_b))

        self.left_walls= self._segment_walls(self.wall_l, 'y')
        self.right_walls= self._segment_walls(self.wall_r, 'y')
        self.top_walls= self._segment_walls(self.wall_t, 'x')
        self.bot_walls= self._segment_walls(self.wall_b, 'x')

    def _segment_walls(self, wall: np.ndarray, dir: str) -> List:
        if dir == 'x':
            static_ax = 0
            move_ax = 1
        else:
            static_ax = 1
            move_ax = 0

        # sort x then sort y
        sorted_indices = np.argsort(wall[move_ax], kind='stable')
        sorted = wall[:, sorted_indices]
        sorted_indices = np.argsort(sorted[static_ax], kind='stable')
        sorted = sorted[:, sorted_indices]

        segments = []
        curr = 0
        for i in range(len(sorted[0])-1):
            curr_next = curr
            if sorted[static_ax][i] != sorted[static_ax][i+1]:
                curr_next = i+1
            elif sorted[static_ax][i] == sorted[static_ax][i+1] and abs(sorted[move_ax][i] - sorted[move_ax][i+1]) > 1:
                curr_next = i+1
            if curr != curr_next:
                segments.append((sorted[static_ax][curr], sorted[move_ax][curr], sorted[move_ax][curr_next-1]))
            curr = curr_next

        if len(sorted[0]) > 0:
            segments.append((sorted[static_ax][curr], sorted[move_ax][curr], sorted[move_ax][len(sorted[0])-1]))
        return segments


    def convolve(self, solid: np.ndarray, kernel: np.ndarray, pad: int = 1) -> np.ndarray:
        (k, div) = kernel
        k_size = len(k)
        height, width = solid.shape
        padded = np.pad(solid, (k_size // 2, k_size // 2), mode='constant', constant_values=pad)

        res = []
        for y in range(height):
            for x in range(width):
                res.append(np.sum(padded[y:k_size + y, x:k_size + x] * k) // div)

        res = np.array(res).reshape((height, width)).astype(int)
        return np.where(res == 1, 1, 0) & solid

    def plot(self, store_at: str):
        conc = np.hstack((self.conc_br, self.conc_bl, self.conc_tl, self.conc_tr))
        conv = np.hstack((self.conv_br, self.conv_bl, self.conv_tl, self.conv_tr))
        walls = np.hstack((self.wall_t, self.wall_b, self.wall_l, self.wall_r))

        rgb = np.zeros((self._height, self._width, 3), dtype=np.ubyte)
        for i in range(len(walls)//2):
            for j in range(len(walls[2*i])):
                y = walls[2*i][j]
                x = walls[2*i+1][j]
                rgb[y, x] = (255, 0, 0)
        for i in range(len(conc)//2):
            y = conc[2*i]
            x = conc[2*i+1]
            rgb[y, x] = (0, 255, 0)
        for i in range(len(conv)//2):
            y = conv[2*i]
            x = conv[2*i+1]
            rgb[y, x] = (255, 255, 0)

        if not store_at.endswith("/"):
            store_at += "/"
        plt_img.imsave(store_at + "boundaries.png", rgb)

    conc_bl = ()
    conc_br = ()
    conc_tl = ()
    conc_tr = ()

    conv_bl = ()
    conv_br = ()
    conv_tl = ()
    conv_tr = ()

    wall_t = ()
    wall_b = ()
    wall_l = ()
    wall_r = ()

    _kernel_conc_bl = (np.array([[1, 1, 16],
                                 [1, 1, 1],
                                 [1, 1, 1]]), 8)
    _kernel_conc_br = (np.array([[16, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]]), 8)
    _kernel_conc_tl = (np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 16]]), 8)
    _kernel_conc_tr = (np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [16, 1, 1]]), 8)

    _kernel_conv_tl = (np.array([[8, 8, 1],
                                 [8, 1, 1],
                                 [1, 1, 1]]), 4)
    _kernel_conv_tr = (np.array([[1, 8, 8],
                                 [1, 1, 8],
                                 [1, 1, 1]]), 4)
    _kernel_conv_br = (np.array([[1, 1, 1],
                                 [1, 1, 8],
                                 [1, 8, 8]]), 4)
    _kernel_conv_bl = (np.array([[1, 1, 1],
                                 [8, 1, 1],
                                 [8, 8, 1]]), 4)

    _kernel_top = (np.array([[1, 1, 1],
                             [1, 1, 1],
                             [12, 12, 12]]), 6)
    _kernel_bot = (np.array([[12, 12, 12],
                             [1, 1, 1],
                             [1, 1, 1]]), 6)
    _kernel_left = (np.array([[1, 1, 12],
                              [1, 1, 12],
                              [1, 1, 12]]), 6)
    _kernel_right = (np.array([[12, 1, 1],
                              [12, 1, 1],
                              [12, 1, 1]]), 6)
