# coding=utf-8
"""
_________________________________________________________________________________________________
                                                                                                 |
Authors: * Ulrich Prestel    <Ulrich.Prestel@protonmail.com>                                     |
       : * Holger Wünsche    <Holger.o.wuensche@t-online.de>                                     |
_________________________________________________________________________________________________|
"""

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import scipy.misc
import imageio
import math
from manta import *
import glob

EMITTER_MARGIN = 2


def generate_sdf_field(np_emitter_mask, res_x, res_y):
    inner_sdf = np_emitter_mask
    outer_sdf = 1 - inner_sdf

    inner_sdf = distance_transform_edt(inner_sdf)
    outer_sdf = distance_transform_edt(outer_sdf)

    # we normalize the field to [-1,1] to accelerate the learning for our CNN
    final_sdf = (outer_sdf - inner_sdf) / (np.sqrt(res_x ** 2 + res_y ** 2))

    return final_sdf


class GeometryGenerator(object):
    def __init__(self, geopath, obstacleGrid, parent, flags, res_x, res_y, FlagObstacle, emitters):
        self.geopath = geopath
        self.obstacleGrid = obstacleGrid
        self.parent = parent
        self.res_x = res_x
        self.res_y = res_y
        self.flags = flags
        self.FlagObstacle = FlagObstacle
        self.emitters = emitters
        self.images = []

        print(imageio.__path__)

        for image_path in glob.glob(self.geopath):
            print(image_path, "<<<<<")
            image = imageio.imread(image_path)
            self.images.append(image)

        self.canvas = np.zeros((res_x, res_y), dtype="f") * 255

    def generate(self, n_obstacles):
        print("generated images", len(self.images))
        for _ in range(0, n_obstacles):
            index = np.random.randint(low=0, high=len(self.images))
            im = self.images[index]
            gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
            gray = 255 - np.rot90(gray(im), 3)
            self.__combine_random(self.canvas, gray)

        # now we make sure that the emitters are not obstructed by the geometry
        for emitter in self.emitters:
            em_shape, radius, val = emitter
            c = em_shape.getCenter()
            a = c.x
            b = c.y
            print(radius)
            x, y = np.ogrid[-a:self.res_x - a, -b:self.res_y - b]
            mask = x ** 2 + y ** 2 <= (radius + EMITTER_MARGIN) ** 2
            self.canvas[mask] = 0

        for i in range(1, self.res_x - 1):
            for j in range(1, self.res_y - 1):
                if self.canvas[i][j] == 255:
                    obs = Sphere(parent=self.parent, center=vec3(i + .5, j + .5, 0), radius=.5)
                    obs.applyToGrid(grid=self.flags, value=self.FlagObstacle)
                    obs.applyToGrid(grid=self.obstacleGrid, value=1)

    def __combine_random(self, shape_a, shape_b):
        # assume shape_b <= shape_a
        pos_h = np.random.randint(low=0, high=self.res_x)
        pos_v = np.random.randint(low=0, high=self.res_y)

        shape_b2 = self.__transform_random(shape_b)

        # pos_v, pos_h = 6, 1  # offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + shape_b2.shape[0], shape_a.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + shape_b2.shape[1], shape_a.shape[1]), 0))

        v_range2 = slice(max(0, -pos_v), min(-pos_v + shape_a.shape[0], shape_b2.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + shape_a.shape[1], shape_b2.shape[1]))

        # shape_a[v_range1, h_range1] += shape_b[v_range2, h_range2]
        if np.random.random() >= .1:
            shape_a[v_range1, h_range1] = self.__combine_union(shape_a[v_range1, h_range1],
                                                               shape_b2[v_range2, h_range2])
        else:
            shape_a[v_range1, h_range1] = self.__combine_subtract(shape_a[v_range1, h_range1],
                                                                  shape_b2[v_range2, h_range2])

    def __combine_union(self, shape_a, shape_b):
        shape_a2 = np.clip(shape_a + shape_b, 0, 255)
        return shape_a2

    def __combine_subtract(self, shape_a, shape_b):
        shape_a2 = np.clip(shape_a - shape_b, 0, 255)
        return shape_a2

    def __transform_random(self, shape):
        c = np.random.randint(low=0, high=3)
        return np.rot90(shape, c)
