# @Time    : 2018/9/27 15:01
# @Author  : Guo Yuanfang
# @File    : logistic.py
# @Software: PyCharm


## 在logistic.py中，将会利用UCI-Ablone数据集，比较自编的逻辑算法与sklearn中逻辑算法。

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm


class LogisticReg:
    def __init__(self):
        pass

    def fit(self, X, y):
        # 随机梯度下降法
        num_sample, num_feature = X.shape
        eta = 0.05  # fixed learning rate
        num_sample, num_feature = X.shape
        one = np.ones((num_sample, 1))
        Xh = np.mat(np.hstack((X, one)))
        yh = np.mat(y).T
        wh = np.mat(np.ones((num_feature + 1, 1)))  # 保存上一个w值
        wn = np.mat(np.random.randn(num_feature + 1, 1))  # 保存最新的w值

        grad = 0
        step = 0
        while np.linalg.norm(wh - wn) > 0.002:
            step += 1

            # print("step{}: {}".format(step, wn))

            sam_selected = np.random.randint(0, num_sample)  # 选择某一条sample求梯度

            xs = np.mat(Xh[sam_selected])
            ys = np.mat(y[sam_selected])

            p1 = np.exp(xs * wn) / (np.exp(xs * wn) + 1)
            grad = - (ys - p1) * xs
            wh = wn
            wn = wn - eta * grad.T

        self.w = wn

    def predict(self, X):
        num_sample, num_feature = X.shape
        one = np.ones((num_sample, 1))
        Xh = np.mat(np.hstack((X, one)))
        p1 = np.exp(Xh * self.w) / (np.exp(Xh * self.w) + 1)
        return (p1 > 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return f1_score(y, y_pred)


def cross_validation_score(estimator, X, y, cv):
    score = []
    kfold = KFold(cv, True)
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score.append(f1_score(y_test, y_pred))

    return score


def leave_one_out(estimator, X, y):
    num_sample, num_feature = X.shape
    y_pred = np.zeros(num_sample)
    for i in range(num_sample):
        X_train_temp, X_test_temp = np.vstack((X[:i], X[i + 1:])), X[i]
        y_train_temp, y_test_temp = np.hstack((y[:i], y[i + 1:])), y[i]
        X_test_temp = X_test_temp.reshape(1, -1)
        estimator.fit(X_train_temp, y_train_temp)
        y_pred[i] = estimator.predict(X_test_temp)
    score = f1_score(y, y_pred)
    return score


def test_10_fold():
    # 第一步，载入数据
    name = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
            'Rings']
    data = pd.read_csv('abalone.data', header=None, names=name)

    # 第二步，预处理数据
    ## 我们发现该数据集中第一个特征为Sex，里面有“M”“F”“I”三种类型，决定用one-hot处理
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = pd.get_dummies(X)

    # 将其变为一个二分类问题
    X = np.vstack((X[y == 8], X[y == 10]))
    y = np.hstack((y[y == 8], y[y == 10]))
    y = (y > 9).astype(int)
    # # 10折交叉验证
    ## 利用sklearn中算法
    lr1 = LogisticRegression()
    scores1 = cross_validation_score(lr1, X, y, cv=10)

    ## 利用自编算法
    lr2 = LogisticReg()
    scores2 = cross_validation_score(lr2, X, y, cv=10)
    print(scores1)
    print(scores2)
    fig = plt.figure(figsize=(9, 6))
    n = 10
    X_1 = np.arange(n) + 1
    plt.bar(X_1, scores1, width=0.35, facecolor='lightskyblue', edgecolor='white')
    plt.bar(X_1 + 0.35, scores2, width=0.35, facecolor='yellowgreen', edgecolor='white')
    fig.legend(['sk-learn', 'self'])
    plt.title("Logistic - 10Fold")
    plt.plot()


def test_leave_one_out():
    # 第一步，载入数据
    name = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
            'Rings']
    data = pd.read_csv('abalone.data', header=None, names=name)

    # 第二步，预处理数据
    ## 我们发现该数据集中第一个特征为Sex，里面有“M”“F”“I”三种类型，决定用one-hot处理
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = pd.get_dummies(X)

    # 将其变为一个二分类问题
    X = np.vstack((X[y == 8], X[y == 10]))
    y = np.hstack((y[y == 8], y[y == 10]))
    y = (y > 9).astype(int)
    # 留一法
    ## 利用sklearn中算法
    lr1 = LogisticRegression()
    scores1 = leave_one_out(lr1, X, y)

    ## 利用自编算法
    lr2 = LogisticReg()
    scores2 = leave_one_out(lr2, X, y)
    print(scores1)
    print(scores2)
    fig = plt.figure(figsize=(9, 6))
    n = 10
    X_1 = np.arange(n) + 1
    plt.bar(1, scores1)
    plt.bar(2, scores2)
    fig.legend(['sk-learn', 'self'])
    plt.title("Logistic - LeaveOneOut")
    plt.plot()


def test_logistics():
    test_10_fold()
    test_logistics()


if __name__ == '__main__':
    test_logistics()
