import random
import numpy as np
from data_preprocessing import data_preprocessing
from train import train
from validation import validation

DATA_PATH = "data.txt"
SPLIT_RATE = 0.7
LEARNING_RATE = 1

if __name__ == '__main__':
    # create dataset:
    #   if x_1 < 0, y = -1; else y=1
    with open(DATA_PATH, "w") as f:
        for i in range(100):
            x_1 = random.uniform(-100, 100)
            x_2 = random.uniform(-100, 100)
            if x_1 < 0:
                y = -1
            else:
                y = 1
            f.write("{} {} {}\n".format(x_1, x_2, y))
    train_x, train_y, val_x, val_y = data_preprocessing(data_path=DATA_PATH, split_rate=SPLIT_RATE).run()
    w, b = train(train_x, train_y, LEARNING_RATE).run()
    acc = validation(val_x, val_y, w, b).compute_acc()
    print(acc)