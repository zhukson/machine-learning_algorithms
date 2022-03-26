import numpy as np


class validation:
    def __init__(self, val_x, val_y, w, b):
        self.val_x = val_x
        self.val_y = val_y
        self.num_dim = len(self.val_x[0])
        self.num_sample = len(self.val_x)
        self.w = w
        self.b = b

    def inference(self, x):
        result = np.dot(self.w, np.reshape(x, (self.num_dim, 1)))
        cls = 1 if result > 0 else -1
        return cls

    def compute_acc(self):
        positive = 0
        for i in range(0, self.num_sample):
            cls = self.inference(self.val_x[i])
            if cls == self.val_y[i]:
                positive += 1
        return positive/self.num_sample
