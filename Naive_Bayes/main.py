import numpy as np
from data_preprocessing import data_preprocessing
from NaiveBayes import NaiveBayes

if __name__ == '__main__':
    train_x, train_y, val_x, val_y = data_preprocessing("data.txt", 1).run()
    t = NaiveBayes(train_x, train_y)
    t.train()
    ref = {0: 1, 1: -1}
    final_result = ref[t.predict([2, 1])]
    print(final_result)
