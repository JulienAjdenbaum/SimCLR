import numpy as np


class AdditiveGaussianNoisePerturbator:
    """
    AdditiveGaussianNoisePerturbator is a pertubator that takes a Volume as input and
    send back an array of data with additive gaussian noise.
    """

    def __init__(self, mean=0.0, std_dev=1.0, prob=1.0):
        self.mean = mean
        self.std_dev = std_dev
        self.prob = prob

    def __call__(self, x):
        if np.random.random() < self.prob:
            return x + np.random.normal(self.mean, self.std_dev, size=x.size)
        else:
            return x
