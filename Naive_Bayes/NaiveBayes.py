import numpy as np
from data_preprocessing import data_preprocessing


class NaiveBayes:

    def __init__(self, x, y, laplacian_smoothing=0.1):
        self.x = x
        self.y = y
        self.cls = set(y)
        self.cls_dict = dict(zip(self.cls, [i for i in range(0, len(self.cls))]))
        self.lp = laplacian_smoothing
        self.NUM_CLS = len(self.cls)
        self.NUM_SAMPLE = len(x)
        self.NUM_ATT = len(x[0])
        self.att_list = self.att()
        self.max_att = 0
        for i in self.att_list:
            self.max_att = max(len(i), self.max_att)
        self.MLE_prior = np.zeros(self.NUM_CLS)
        self.MLE_post = np.zeros([self.NUM_ATT, self.max_att, self.NUM_CLS])
        self.att2index = [{} for i in range(self.NUM_ATT)]
        for i in range(self.NUM_ATT):
            val = range(len(self.att_list[i]))
            key = self.att_list[i]
            self.att2index[i] = dict(zip(key, val))

    def att(self):
        att = [set() for i in range(self.NUM_ATT)]
        for i in range(self.NUM_SAMPLE):
            for j in range(self.NUM_ATT):
                att[j].add(self.x[i][j])
        return att

    # MLE先验概率
    # input: class
    # output: prior probability: P(Y = cls)
    def compute_prior(self, cls):
        count = 0
        for i in range(self.NUM_SAMPLE):
            if self.y[i] == cls:
                count += 1
        prior = (count + self.lp) / (self.NUM_SAMPLE + (self.NUM_SAMPLE * self.lp))
        return prior

    def generate_prior_matrix(self):
        for i, cls in enumerate(self.cls_dict):
            self.MLE_prior[i] = self.compute_prior(cls)

    # MLE条件概率
    # input: attribute_dim, attribute_val, class
    # output: posterior probability: P(X = aji| Y = cls)
    def compute_post(self, j, a, cls):
        count_comb = 0
        for i in range(self.NUM_SAMPLE):
            if self.x[i][j] == a and self.y[i] == cls:
                count_comb += 1
        # 获取prior matrix所计算过的先验概率
        index = self.cls_dict[cls]
        post = (count_comb + self.lp) / (self.MLE_prior[index] * (self.NUM_SAMPLE + (self.NUM_SAMPLE * self.lp))
                                         - self.lp + len(self.att_list[j]) * self.lp)
        return post

    def generate_post_matrix(self):
        # dim of attribute
        for j in range(self.NUM_ATT):
            # attribute val
            for a_jl_index, a_jl in enumerate(self.att_list[j]):
                # class
                for cls_index, cls in enumerate(self.cls):
                    self.MLE_post[j][a_jl_index][cls_index] = self.compute_post(j, a_jl, cls)

    def train(self):
        self.generate_prior_matrix()
        self.generate_post_matrix()

    def predict(self, x):
        result = np.ones(self.NUM_CLS)
        for j, val in enumerate(x):
            a_jl_index = self.att2index[j][val]
            for cls in self.cls:
                cls_index = self.cls_dict[cls]
                result[cls_index] *= self.MLE_post[j][a_jl_index][cls_index]
        return np.argmax(result)