import random
import numpy as np
from data_preprocessing import data_preprocessing
from Perceptron import Perceptron

DATA_PATH = "data.txt"
SPLIT_RATE = 1
LEARNING_RATE = 0.01

if __name__ == '__main__':
    train_x, train_y, val_x, val_y = data_preprocessing(data_path=DATA_PATH, split_rate=SPLIT_RATE).run()
    p1 = Perceptron(train_x, train_y, LEARNING_RATE)
    p1.train()
    p1.plot()
