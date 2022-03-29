import numpy as np
from Node import Node


# input: training data, label, NUM_neighbors


class kd_tree:

    def __init__(self, x, y, k):
        self.current_dim = 0
        self.x = x
        self.y = y
        self.k = k
        self.NUM_DIM = len(x[0])
        self.NUM_CLS = len(set(y))
        self.target = None
        self.nearest_distance = [10000000 for i in range(self.k)]
        self.nearest_node = [0 for i in range(self.k)]
        self.visited_node = []

    # using training data to construct the kd_tree
    # input: samples x, label, current dimension, root_node
    # output: the root_node of a constructed kd-tree
    def construct(self, x, label, dim, parent_node):
        if len(x) == 0:
            return None
        sorted_x, sorted_label, median_index = self.compute_median(x, label, dim)
        current_Node = Node(sorted_x[median_index], label[median_index], dim)
        current_Node.parent = parent_node
        current_Node.left = self.construct(sorted_x[:median_index], sorted_label[:median_index],
                                           (dim + 1) % self.NUM_DIM, current_Node)
        current_Node.right = self.construct(sorted_x[median_index + 1:], sorted_label[median_index + 1:],
                                            (dim + 1) % self.NUM_DIM, current_Node)
        return current_Node

    # sort and compute the median of x and y based on the current dimension.
    def compute_median(self, x, y, dim):
        sorted_x = sorted(x, key=lambda i: i[dim])
        sorted_y = sorted(y)
        median_index = int(len(sorted_x) / 2)
        return sorted_x, sorted_y, median_index

    # searching the knn of the target sample and save them into self.nearest_node.
    # input: root_node, current_dimension
    # output: None

    def search(self, current_node, dim):
        if current_node is None or current_node in self.visited_node:
            return
        self.visited_node.append(current_node)
        # compute distance and update
        current_distance = self.euclidean_distance(current_node.x, self.target)
        self.update_distance(current_distance, current_node)
        # keep move downward
        if current_node.x[dim] >= self.target[dim]:
            self.search(current_node.left, (dim + 1) % self.NUM_DIM)
        else:
            self.search(current_node.right, (dim + 1) % self.NUM_DIM)
        # if the dim val difference between current node and target is smaller than nearest distance, move to the
        # other child node. otherwise, move back to the parent's node.
        if current_node.parent is None:
            return
        if abs(current_node.x[dim] - self.target[dim]) < self.nearest_distance[-1]:
            if current_node.parent.left.x == current_node.x:
                self.search(current_node.parent.right, (dim + 1) % self.NUM_DIM)
            else:
                self.search(current_node.parent.left, (dim + 1) % self.NUM_DIM)

    # update the nearest distance and nearest node.
    def update_distance(self, current_distance, current_node):
        for i, val in enumerate(self.nearest_distance):
            if current_distance < val:
                self.nearest_distance.insert(i, current_distance)
                self.nearest_node.insert(i, current_node)
                self.nearest_distance.pop()
                self.nearest_node.pop()
                break

    # compute the euclidean distance
    def euclidean_distance(self, a, b):
        a, b = np.array(a), np.array(b)
        return np.sqrt(np.sum(np.square(a - b)))

    # constructing the kd-tree and searching knn of the target.
    # input: target
    # output: the knn of target sample.
    def pred(self, target):
        root_node = self.construct(x=self.x, label=self.y, dim=0, parent_node=None)
        self.target = target
        self.search(root_node, 0)
        return self.nearest_node
