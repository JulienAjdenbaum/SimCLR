import numpy as np
from scipy.ndimage import gaussian_filter


class GaussianBlurPerturbator:
    def __init__(self, sigma=1.0, prob=1.0, sigma_range=None):
        self.sigma = sigma
        self.prob = prob
        self.sigma_range = sigma_range

    def __call__(self, x):
        if np.random.random() < self.prob:
            if self.sigma_range:
                self.sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            return gaussian_filter(x, self.sigma)
        else:
            return x
