import numpy as np


class train:
    def __init__(self, train_x, train_y, learning_rate):
        self.train_x = train_x
        self.train_y = train_y
        self.num_sample = len(train_x)
        self.num_dim = len(train_x[0])
        self.w = np.ones((1, self.num_dim))
        self.b = 0
        self.learning_rate = learning_rate

    # gradient_decent: d/dw = yx; d/db = y
    #                  w += learning_rate*yx
    #                  b += learning_rate*y

    def gradient_decent(self, x, y):
        for i in range(len(x)):
            self.w[0][i] += self.learning_rate * x[i] * y
        self.b += self.learning_rate * y

    def run(self):
        for i in range(0, len(self.train_x)):
            current_x = np.array(self.train_x[i])
            current_y = self.train_y[i]
            # Loss = y*sing(wx+b)
            current_loss = current_y * np.dot(self.w, current_x.T) + self.b
            if current_loss < 0:
                self.gradient_decent(current_x, current_y)
        return self.w, self.b
