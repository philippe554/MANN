import numpy as np
import random

class Data:
    def __init__(self, X, Y, C):
        self.X = X
        self.Y = Y
        self.C = C

    def getBatch(self, amount):
        indices = sorted(random.sample(range(len(self.X)), amount))
        x = [self.X[i] for i in indices]
        y = [self.Y[i] for i in indices]

        return x, y

