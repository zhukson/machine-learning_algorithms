import numpy as np


class Node:

    def __init__(self, x, y, dim):
        self.dim = dim
        self.x = x
        self.y = y
        self.left = None
        self.right = None
        self.parent = None