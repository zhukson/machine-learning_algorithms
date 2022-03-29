import numpy as np
from data_preprocessing import data_preprocessing
from logistic_regression import logistic_regression

LEARNING_RATE = 0.1
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_x, train_y, val_x, val_y = data_preprocessing("data.txt", 0.7).run()
    lr = logistic_regression(train_x, train_y, val_x, val_y, LEARNING_RATE)
    lr.train()
    print("validation_acc: {}".format(lr.validation()))
    lr.plot()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
