import random

from scipy import ndimage


class Rotation:
    def __init__(self, random_angle=True, angle=0.0, prob=1.0):
        self.random_angle = random_angle
        self.angle = angle
        self.prob = prob

    def __call__(self, x):
        if random.random() < self.prob:
            if self.random_angle:
                return ndimage.rotate(x, 360 * random.random(), reshape=False)
            return ndimage.rotate(x, self.angle, reshape=False)
        return x
