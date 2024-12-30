# modified from https://github.com/danfergo/gelsight_simulation

import time
import os
import numpy as np
import cv2
import numpy as np
import scipy.ndimage.filters as fi
import math
from omegaconf import OmegaConf


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def gaussian_noise(image, sigma):
    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def derivative(mat, direction):
    assert (
        direction == "x" or direction == "y"
    ), "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == "x":
        kernel = [[-1.0, 0.0, 1.0]]
    elif direction == "y":
        kernel = [[-1.0], [0.0], [1.0]]
    kernel = np.array(kernel, dtype=np.float64)
    return cv2.filter2D(mat, -1, kernel) / 2.0


def tangent(mat):
    dx = derivative(mat, "x")
    dy = derivative(mat, "y")
    img_shape = np.shape(mat)
    _1 = (
        np.repeat([1.0], img_shape[0] * img_shape[1])
        .reshape(img_shape)
        .astype(dx.dtype)
    )
    unormalized = cv2.merge((-dx, -dy, _1))
    norms = np.linalg.norm(unormalized, axis=2)
    return unormalized / np.repeat(norms[:, :, np.newaxis], 3, axis=2)


def solid_color_img(color, size):
    image = np.zeros(size + (3,), np.float64)
    image[:] = color
    return image


def add_overlay(rgb, alpha, color):
    s = np.shape(alpha)

    opacity3 = np.repeat(alpha, 3).reshape((s[0], s[1], 3))

    overlay = solid_color_img(color, s)

    foreground = opacity3 * overlay
    background = rgb.astype(np.float64)
    res = background + foreground

    res[res > 255.0] = 255.0
    res[res < 0.0] = 0.0
    res = res.astype(np.uint8)

    return res


class PhongShadingRenderer:

    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "render_config.yml")
        config = OmegaConf.load(config_path).sensor
        self.light_sources = config["light_sources"]
        self.with_background = (
            config["with_background"] if "with_background" in config else False
        )
        self.background = cv2.imread(
            os.path.join(os.path.dirname(__file__), config["background_img"])
        )
        self.px2m_ratio = config["px2m_ratio"]
        self.elastomer_thickness = config["elastomer_thickness"]
        self.max_depth = config["max_depth"]

        self.default_ks = 0.15
        self.default_kd = 0.5
        self.default_alpha = 5

        self.ka = config["ka"] or 0.8

        self.enable_depth_texture = (
            config["enable_depth_texture"]
            if "enable_depth_texture" in config
            else False
        )
        self.texture_sigma = (
            config["texture_sigma"] if "texture_sigma" in config else 0.000001
        )
        self.t = config["t"] if "t" in config else 3
        self.sigma = config["sigma"] if "sigma" in config else 7
        self.kernel_size = config["kernel_size"] if "kernel_size" in config else 21
        self.enable_shadow = (
            config["enable_shadow"] if "enable_shadow" in config else False
        )

        self.min_depth = self.max_depth - self.elastomer_thickness
        self.background_color = self._get_background_color()

    def _get_shadow_map(self, light_dir):
        (w, h) = self.depth.shape

        depth_buffer_resolution = self.px2m_ratio * 1000
        threshold = depth_buffer_resolution / (
            light_dir[2] / math.sqrt(light_dir[0] ** 2 + light_dir[1] ** 2)
        )

        new_z = light_dir / np.linalg.norm(light_dir)

        try_x = np.cross(np.array([0, 1, 0]), new_z)
        if np.linalg.norm(try_x) != 0:
            new_x = try_x / np.linalg.norm(try_x)
            new_y = np.cross(new_z, new_x)
        else:
            try_y = np.cross(new_z, np.array([1, 0, 0]))
            new_y = try_y / np.linalg.norm(try_y)
            new_x = np.cross(new_y, new_z)

        transform_mat = np.dstack((new_x, new_y, new_z))

        new_points = self.points @ transform_mat

        new_points_x_min = np.min(new_points[:, :, 0])
        new_points_x_max = np.max(new_points[:, :, 0])
        new_points_y_min = np.min(new_points[:, :, 1])
        new_points_y_max = np.max(new_points[:, :, 1])

        buffer_w = (
            math.ceil((new_points_x_max - new_points_x_min) / depth_buffer_resolution)
            + 1
        )
        buffer_h = (
            math.ceil((new_points_y_max - new_points_y_min) / depth_buffer_resolution)
            + 1
        )
        buffer = (
            np.ones((buffer_w, buffer_h)).astype(np.float32) * 1e20
        )  # to record the min depth
        shadow_map = np.ones((w, h))

        for i in range(w):
            for j in range(h):
                new_point = new_points[i, j, ...]
                coord_x_in_buffer = math.floor(
                    (new_point[0] - new_points_x_min) / depth_buffer_resolution
                )
                coord_y_in_buffer = math.floor(
                    (new_point[1] - new_points_y_min) / depth_buffer_resolution
                )
                if (
                    new_point[2] < buffer[coord_x_in_buffer, coord_y_in_buffer]
                ):  # need to update
                    buffer[coord_x_in_buffer, coord_y_in_buffer] = new_point[
                        2
                    ]  # store the depth value

        coord_x_in_buffer = np.floor(
            (new_points[:, :, 0] - new_points_x_min) / depth_buffer_resolution
        ).astype(np.int)
        coord_y_in_buffer = np.floor(
            (new_points[:, :, 1] - new_points_y_min) / depth_buffer_resolution
        ).astype(np.int)
        in_shadow = (
            new_points[..., 2]
            > buffer[coord_x_in_buffer, coord_y_in_buffer] + threshold
        )
        shadow_map[in_shadow] = 0  # need to update

        end = time.time()

        kernel = gkern2(15, 7)
        shadow_map = cv2.filter2D(shadow_map, -1, kernel)

        return shadow_map

    def _get_background_color(self):
        depth = np.zeros((10, 10))
        background_rendered = self._generate(depth, noise=False)
        return background_rendered.mean(axis=(0, 1))

    def _phong_illumination(self, depth, T, light_dir, kd, ks, alpha):

        dot = np.dot(T, np.array(light_dir)).astype(np.float64)
        diffuse_l = dot * kd
        diffuse_l[diffuse_l < 0] = 0.0

        dot3 = np.repeat(dot[:, :, np.newaxis], 3, axis=2)

        R = 2.0 * dot3 * T - light_dir
        V = [0.0, 0.0, 1.0]

        spec_l = np.power(np.dot(R, V), alpha) * ks

        if self.enable_shadow:
            shadow_map = self._get_shadow_map(light_dir)
            return (diffuse_l + spec_l) * shadow_map
        else:
            return diffuse_l + spec_l

    def _generate(self, target_depth, noise=False):

        if noise:
            textured_elastomer_depth = gaussian_noise(
                target_depth, self.texture_sigma
            )  # add noise
        else:
            textured_elastomer_depth = target_depth.copy()

        out = np.zeros((target_depth.shape[0], target_depth.shape[1], 3))

        self.depth = target_depth
        depth = (self.depth * 1000).copy()
        (h, w) = depth.shape
        x = np.linspace(0, (w - 1) * self.px2m_ratio * 1000, num=w)
        y = np.linspace(0, (h - 1) * self.px2m_ratio * 1000, num=h)

        xx, yy = np.meshgrid(x, y)
        self.points = np.dstack((xx, yy, depth))

        T = tangent(textured_elastomer_depth / self.px2m_ratio)

        # show_normalized_img('tangent', T)
        for light in self.light_sources.values():
            ks = light["ks"] if "ks" in light else self.default_ks
            kd = light["kd"] if "kd" in light else self.default_kd
            alpha = light["alpha"] if "alpha" in light else self.default_alpha
            out = add_overlay(
                out,
                self._phong_illumination(
                    target_depth, T, light["position"], kd, ks, alpha
                ),
                light["color"],
            )

        return out

    def generate(self, target_depth, return_depth=False):
        out = self._generate(target_depth, noise=self.enable_depth_texture)
        if self.with_background:
            diff = (
                out.astype(np.float32)
                - solid_color_img(self.background_color, out.shape[:2])
            ) * 1
            kernel = gkern2(5, 2)
            diff = cv2.filter2D(diff, -1, kernel)

            # Combine the simulated difference image with real background image
            result = self.ka * self.background.copy()

            result = np.clip((diff + result), 0, 255).astype(np.uint8)
        else:
            result = out
        if return_depth:
            return result, target_depth
        else:
            return result
