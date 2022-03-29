import numpy as np
import matplotlib.pyplot as plt


class logistic_regression:

    def __init__(self, train_x, train_y, val_x, val_y, learning_rate):
        self.train_x, self.train_y = np.array(train_x), np.array(train_y)
        self.val_x, self.val_y = np.array(val_x), val_y
        self.m, self.n = np.shape(self.train_x)
        self.cls = set(train_y)
        self.w = np.ones([1, self.n])
        self.learning_rate = learning_rate

    # put all of the samples into gradient decent.
    # time complexity O(epoch * m * n)
    def gradient_decent(self, x, y):
        EPOCH = 100
        for i in range(EPOCH):
            h = self.sigmoid(np.dot(self.w, x.T))
            dw = np.dot((y - h), x)
            self.w -= -1 * self.learning_rate * dw

    # only pick one sample for gradient decent at a time.
    # time complexity O(epoch * m)
    def sgd(self, x, y):
        EPOCH = 5
        for epoch in range(EPOCH):
            for i in range(self.m):
                # (1,3)x(3*1) = (1,1)
                h = self.sigmoid(np.dot(self.w, x[i].T))
                dw = np.dot((y[i] - h).reshape(1, 1), x[i].reshape(1, 3))
                self.w -= -1 * self.learning_rate * dw
            self.learning_rate /= 2

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def train(self):
        self.sgd(self.train_x, self.train_y)

    def validation(self):
        positive = 0
        for i in range(len(self.val_x)):
            prob = self.sigmoid(np.dot(self.w, self.val_x[i].T))
            cls = 1 if prob >= 0.5 else 0
            if cls == self.val_y[i]:
                positive += 1
        return positive / len(self.val_x)

    def plot(self):
        # make the data
        np.random.seed(3)
        x_0, y_0, x_1, y_1 = [], [], [], []
        for i, sample in enumerate(self.val_x):
            if self.val_y[i] == 0:
                x_0.append(sample[0])
                y_0.append(sample[1])
            elif self.val_y[i] == 1:
                x_1.append(sample[0])
                y_1.append(sample[1])

        # size and color:
        sizes = 10
        # plot
        fig, ax = plt.subplots()
        ax.scatter(x_0, y_0, s=sizes, vmin=0, vmax=100)
        ax.scatter(x_1, y_1, s=sizes, vmin=0, vmax=100)
        ax.set(xlim=(0, 5), xticks=np.arange(-5, 5),
               ylim=(0, 5), yticks=np.arange(-5, 20))
        x = np.arange(-4, 4, 0.1)
        y = (-1 * self.w[0][2] - self.w[0][0] * x) / self.w[0][1]
        ax.plot(x, y, c='red')
        plt.show()
