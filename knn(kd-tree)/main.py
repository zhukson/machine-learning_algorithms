import numpy as np
from data_preprocessing import data_preprocessing
from kd_tree import kd_tree

# the sample you want to do classification
TARGET = [9, 5]

if __name__ == '__main__':
    train_x, train_y, _, _ = data_preprocessing("data.txt", 1).run()
    tree = kd_tree(train_x, train_y, 3).pred(TARGET)
    cls2index = {1: 0, 2: 1}
    index2cls = [1, 2]
    count_cls = [0] * 2
    for i, node in enumerate(tree):
        print("nearest node {}: {} with label {}".format(i, node.x, node.y))
        count_cls[cls2index[node.y]] += 1
    print("classification result of sample {} : {}".format(TARGET, index2cls[np.argmax(count_cls)]))
