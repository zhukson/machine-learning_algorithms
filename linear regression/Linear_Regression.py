import numpy as np
import matplotlib.pyplot as plt


class Linear_Regression:

    def __init__(self, train_x, train_y, val_x, val_y, learning_rate):
        self.train_x, self.train_y = np.array(train_x), np.array(train_y).reshape(len(train_y), 1)
        self.val_x, self.val_y = np.mat(val_x), np.mat(val_y)
        self.m, self.n = len(train_x), len(train_x[0])
        self.w = np.ones([self.n, 1])
        # self.b = 1
        self.learning_rate = learning_rate

    # matrix computation method
    def gradient_decent(self):
        n1 = np.dot(self.train_x, self.w) - self.train_y
        dw = self.learning_rate * 2 * np.dot(self.train_x.T, n1) / 2
        self.w -= dw

    # for loop method
    # def gradient_decent(self):
    #     # print(self.train_x.shape, self.w.shape, self.train_y.shape)
    #     for i in range(self.m):
    #         x, y = self.train_x[i], self.train_y[i]
    #         dw = 2 * (self.w * x + self.b - y) * self.w
    #         db = 2 * (self.w * x + self.b - y)
    #         self.w -= self.learning_rate * dw
    #         self.b -= self.learning_rate * db

    def plot(self):
        fig, ax = plt.subplots()
        x = []
        for i in self.train_x:
            x.append(i[0])
        plt.scatter(self.train_y, x)
        x = np.arange(-8, 8, 0.1)
        y = []
        for i in x:
            y.append(self.w[0] * i + self.w[1])
        plt.plot(y, x, c='red')
        ax.set(xlim=(0, 5), xticks=np.arange(-1, 7),
               ylim=(0, 5), yticks=np.arange(-1, 7))
        plt.show()
