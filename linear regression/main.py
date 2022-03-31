from data_preprocessing import data_preprocessing
from Linear_Regression import Linear_Regression
import numpy as np

if __name__  ==  "__main__":
    train_x, train_y, val_x, val_y = data_preprocessing("data.txt", 1).run()
    lr = Linear_Regression(train_x, train_y, val_x, val_y, 0.01)
    lr.gradient_decent()
    lr.plot()