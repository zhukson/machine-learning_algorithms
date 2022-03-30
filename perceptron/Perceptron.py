import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, train_x, train_y, learning_rate):
        self.train_x = train_x
        self.train_y = train_y
        self.num_sample = len(train_x)
        self.num_dim = len(train_x[0])
        self.w = np.ones((1, self.num_dim))
        self.b = 1
        self.learning_rate = learning_rate
        self.y_dict = {0: -1, 1: 1}

    # gradient_decent: d/dw = yx; d/db = y
    #                  w += learning_rate*yx
    #                  b += learning_rate*y

    def gradient_decent(self, x, y):
        self.w += self.learning_rate * y * x
        self.b += self.learning_rate * y

    def train(self):
        max_iteration = 1000
        for epoch in range(max_iteration):
            Flag = True
            for i in range(0, len(self.train_x)):
                current_x = np.array(self.train_x[i])
                current_y = self.y_dict[self.train_y[i]]
                # Loss = y*sing(wx+b)
                # print(current_y)
                inference_result = np.dot(self.w, current_x.T) + self.b
                current_loss = current_y * inference_result
                if current_loss < 0:
                    Flag = False
                    self.gradient_decent(current_x, current_y)
            if Flag:
                break

    def inference(self, x):
        result = np.dot(self.w, np.array(x).T)+self.b
        cls = 1 if result > 0 else -1
        return cls

    def plot(self):
        x1, y1, x2, y2 = [], [], [], []
        for i in range(0, self.num_sample):
            if self.y_dict[self.train_y[i]] == 1:
                x1.append(self.train_x[i][0])
                y1.append(self.train_x[i][1])
            else:
                x2.append(self.train_x[i][0])
                y2.append(self.train_x[i][1])
        fig, ax = plt.subplots()
        plt.scatter(x1, y1)
        plt.scatter(x2, y2)
        x = np.arange(-4, 4, 0.1)
        # w1*x1+w2*x2+b = 0
        # x2 = -b-w1*x1 / w2
        y = (-1 * self.w[0][0] * x - self.b) / self.w[0][1]
        ax.plot(x, y, c='red')
        ax.set(xlim=(0, 5), xticks=np.arange(-5, 5),
               ylim=(0, 5), yticks=np.arange(-5, 20))
        plt.show()
