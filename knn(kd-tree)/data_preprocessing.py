import numpy as np
import os
import pandas as pd


class data_preprocessing:
    def __init__(self, data_path, split_rate):
        self.data_path = data_path
        self.split_rate = split_rate

    def get_data(self):
        data = np.genfromtxt(self.data_path, dtype = [int, int, int])
        x = []
        y = []
        for i in data:
            c_x = []
            for j in range(len(i)-1):
                c_x.append(i[j])
            x.append(c_x)
            y.append(i[-1])
        return x, y

    def split(self, x, y):
        end_index = len(x) * self.split_rate
        train_x = x[:int(end_index)]
        train_y = y[:int(end_index)]
        val_x = x[int(end_index):]
        val_y = y[int(end_index):]
        return train_x, train_y, val_x, val_y

    def run(self):
        x, y = self.get_data();
        return self.split(x, y)