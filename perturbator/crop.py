import random

import numpy as np
from skimage import transform


class Crop:
    def __init__(self, prob=1.0, crop_size_ratio=(1 / 4, 1 / 4), cropping_type="center",
                 resize=False, keep_dim=False, crop_start=None):
        self.prob = prob
        self.crop_size_ratio = crop_size_ratio
        self.cropping_type = cropping_type
        self.resize = resize
        self.keep_dim = keep_dim
        self.crop_start = crop_start

    def __call__(self, x):
        arr = x
        if random.random() >= self.prob:
            return x
        self.perturbation_used = 1
        assert self.cropping_type in ["center", "random"]
        assert isinstance(arr, np.ndarray)

        img_shape = np.array(arr.shape)
        indexes, crop_shape = self.generate_indexes(arr, img_shape, self.cropping_type)
        if self.resize:
            # resize the image to the input shape
            return transform.resize(arr[tuple(indexes)], img_shape, preserve_range=True)

        if self.keep_dim:
            arr_copy = self.get_cropped_array(arr, img_shape, indexes)
            if np.sum(arr_copy != 0) / np.prod(crop_shape) <= 0.5:
                indexes, crop_shape = self.generate_indexes(arr, img_shape, "center")
                return self.get_cropped_array(arr, img_shape, indexes)
            return arr_copy

        return arr[tuple(indexes)]

    @staticmethod
    def get_cropped_array(arr, img_shape, indexes):
        mask = np.zeros(img_shape, dtype=np.bool)
        mask[tuple(indexes)] = True
        arr_copy = arr.copy()
        arr_copy[~mask] = 0
        return arr_copy

    def generate_indexes(self, arr, img_shape, cropping_type):
        if not isinstance(self.crop_size_ratio, list) and not isinstance(
                self.crop_size_ratio, tuple):
            self.crop_size_ratio = [self.crop_size_ratio] * len(img_shape)
        if len(self.crop_size_ratio) != len(img_shape):
            self.crop_size_ratio = [1, *self.crop_size_ratio]
        size = (img_shape * self.crop_size_ratio[:len(img_shape)]).astype(int)
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.crop_start is None:
                if cropping_type == "center":
                    delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
                elif cropping_type == "random":
                    delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
                else:
                    raise ValueError(
                        f"Type should be in ['center', 'random'] but {self.cropping_type} "
                        f"was "
                        f"provided")
            else:
                delta_before = int(self.crop_start[ndim] * img_shape[ndim])
            indexes.append(slice(delta_before, delta_before + size[ndim]))
        return indexes, size
