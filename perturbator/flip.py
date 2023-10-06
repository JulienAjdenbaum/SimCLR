import random

import numpy as np


class Flip:
    def __init__(self, prob=1.0, random_flip=True, axis=(0, -2)):

        self.prob = prob
        self.random_flip = random_flip
        self.axis = axis

    def __call__(self, x):
        cnt = 2
        y = x.copy()
        if self.random_flip:
            while random.random() < self.prob and cnt > 0:
                self.used_perturbation = True
                self.perturbation_used = 1
                degree = random.choice([0, 1])
                y = np.flip(y, axis=degree)
                cnt = cnt - 1
            if cnt == 2:
                self.used_perturbation = False
        else:
            if random.random() < self.prob:
                self.used_perturbation = True
                self.perturbation_used = 1
                for axis in self.axis:
                    if axis != -2:
                        y = np.flip(y, axis=axis)
            else:
                self.used_perturbation = False
        return y
