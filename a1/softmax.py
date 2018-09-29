# @Time    : 2018/9/27 17:04
# @Author  : Guo Yuanfang
# @File    : softmax.py
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


class Softmax:
    def __init__(self):
        pass

    def fit(self, X, y):
        num_sample, num_feature = X.shape
        labels = list(set(y))
        num_label = len(labels)
        label2int = {}
        for i in range(num_label):
            label2int[labels[i]] = i

        self.labels = labels
        self.label2int = label2int
        num_sample, num_feature = X.shape
        one = np.ones((num_sample, 1))
        Xh = np.mat(np.hstack((X, one)))
        yh = np.mat(y).T
        W = np.mat(np.ones((num_label, num_feature + 1)))  # 保存上一个w值
        Wn = np.mat(0.5 * np.random.randn(num_label, num_feature + 1))  # 保存最新的w值
        step = 0
        while np.linalg.norm(Wn - W) > 0.00001 or step < 40:
            step += 1
            eta = 0.5 / step
            # print("step{}: {}".format(step, Wn))
            grads = 0
            for i in range(10):
                sam_selected = np.random.randint(0, num_sample)
                xs = np.mat(Xh[sam_selected])
                ys = y[sam_selected]

                p = Wn * xs.T
                p = np.exp(p)
                p = p / p.sum()

                err = np.zeros((num_label, 1))
                err[label2int[ys]] = 1
                err = err - p  # k * 1
                errtemp = np.array(err).reshape(num_label)  # 形成一个一维数组以便于diag函数
                err = np.diag(errtemp)  # k * k

                ones = np.mat(np.ones((num_label, 1)))
                Xtemp = ones * xs  # k*(n+1) = (k * 1)*(1*(n+1))

                grads = grads - err * Xtemp  # k*(n+1)
            grad = grads / 10
            W = Wn
            Wn = Wn - eta * grad

        self.W = Wn

    def predict(self, X):
        num_sample, num_feature = X.shape
        one = np.ones((num_sample, 1))
        Xh = np.mat(np.hstack((X, one)))

        p = self.W * Xh.T
        p = np.exp(p)

        y_pred = p.argmax(axis=0)
        ytemp = np.zeros_like(y_pred)
        for i in range(num_sample):
            ytemp[0, i] = self.labels[y_pred[0, i]]  # y_pred数组中存的是0，1，2（label的位置
            # 而y中为相应的label，也就是'F','M'等。
        y_out = np.array(ytemp).reshape(num_sample)
        return y_out


iris = load_iris()
breast_canser = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# softmax
sm = Softmax()
sm.fit(X_train, y_train)
y_pred_sm = sm.predict(X_train)

lr = LogisticRegression()
# one-vs-one
ovo = OneVsOneClassifier(lr)
ovo.fit(X_train, y_train)
y_pred_ovo = ovo.predict(X_test)
# one-vs-rest
ovr = OneVsRestClassifier(lr)
ovr.fit(X_train, y_train)
y_pred_ovr = ovr.predict(X_test)


print("SOFTMAX")
print(classification_report(y_train, y_pred_sm))
print("ONE-VS-ONE")
print(classification_report(y_test, y_pred_ovo))
print("ONE-VS-REST")
print(classification_report(y_test, y_pred_ovr))

print("Softmax f1值为：{}".format(round(f1_score(y_test, y_pred_sm, average='weighted')), 2))
print("one-vs-One f1值为：{}".format(round(f1_score(y_test, y_pred_ovo, average='weighted')), 2))
print("one-vs-Rest f1值为：{}".format(round(f1_score(y_test, y_pred_ovr, average='weighted')), 2))

# f1_sm = []
# f1_ovo = []
# f1_ovr = []
# plot_x = np.linspace(0.1, 0.9, 9)
# for test_rate in plot_x:
#     X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_rate)
#     sm.fit(X_train,y_train)
#     y_pred_sm = sm.predict(X_test)
#     ovo.fit(X_train, y_train)
#     y_pred_ovo = ovo.predict(X_test)
#     ovr.fit(X_train, y_train)
#     y_pred_ovr = ovr.predict(X_test)
#
#
#     f1_sm.append(f1_score(y_test, y_pred_sm, average='weighted'))
#     f1_ovo.append(f1_score(y_test, y_pred_ovo, average='weighted'))
#     f1_ovr.append(f1_score(y_test, y_pred_ovr, average='weighted'))
#
# fig = plt.figure(figsize=(9, 6))
# plt.plot(plot_x[:-1],f1_sm[:-1],'g')
# plt.plot(plot_x[:-1], f1_ovo[:-1], 'r')
# plt.plot(plot_x[:-1], f1_ovr[:-1], 'b')
# fig.legend(['softmax','one-vs-one', 'one-vs-rest'])
# plt.show()
