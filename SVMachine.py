# @Time    : 2018/9/19 13:46
# @Author  : Guo Yuanfang
# @File    : SVMachine.py
# @Software: PyCharm


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


class SVMachine:
    def __init__(self, kernal='linear'):
        #  kernal can be:
        ##  "linear": default
        ##  "poly":
        ##  "rbf": Gaussian radial basis function
        ##  "gauss": Gaussian
        ##  "sigm":  Sigmoid
        self.kernal = kernal

    def fit(self, X, y):
        num_sample, num_feature = X.shape

        # get the Kernal Matrix
        K = np.zeros((num_sample, num_sample))
        if self.kernal == 'linear':
            for i in range(num_sample):
                for j in range(num_sample):
                    K[i, j] = np.dot(X[i], X[j])
        elif self.kernal == 'poly':
            # K(v1,v2)=(γ<v1,v2>+c)^d
            d = 3
            gamma = 1
            c = 1
            for i in range(num_sample):
                for j in range(num_sample):
                    K[i, j] = (gamma * np.dot(X[i], X[j])+c)**d
        elif self.kernal == 'rbf':
            # K(v1,v2)=exp(−γ||v1−v2||2)
            gamma = 1
            for i in range(num_sample):
                for j in range(num_sample):
                    K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j])**2)
        elif self.kernal == 'sigm':
            # K(v1,v2)=tanh(γ<v1,v2>+c)
            gamma = 1
            c = 1
            for i in range(num_sample):
                for j in range(num_sample):
                    K[i, j] = np.tanh(gamma * np.dot(X[i], X[j]) + c)
        else:
            print("Not exists such kernal function! —— {}".format(self.kernal))

        # 利用SMO算法解出alpha

        a = np.ones((num_sample))  # 倒数第二个参数 alpha

        # Initialize alpha
        an = np.random.randn(num_sample)
        t = np.argwhere(y != 0)[0][0]

        an[t] = -(np.dot(an[:t], y[:t]) + np.dot(an[t + 1:], y[t + 1:])) / y[t]

        step = 0
        while True:
            if step > 0 and np.linalg.norm(an - a) < 0.0001:
                print(an)
                print(a)
                break
            step += 1
            print("step{}: {}".format(step, an))
            a = an.copy()

            # 选择某两个下标 进行迭代
            while True:
                i, j = np.random.randint(0, num_sample, 2)
                if i != j and y[j] != 0 and y[i]!=0:
                    break

            c0 = y[i] ** 2 * K[i, i]
            c1 = y[i] * y[j] * K[i, j]
            c2 = y[j] ** 2 * K[j, j]
            c3 = - y[i] / y[j]

            # update alpha[i], alpha[j]
            an[i] = (1 + c3) / (c0 + c1 * c3 + c2 * c3 * c3)
            an[j] = c3 * an[i]

        print("\n核函数为{}\tSMO算法共迭代{}次得到答案w:\n {}".format(self.kernal, step, an))

        self.sample = num_sample
        self.num_feature = num_feature
        self.K = K
        self.alpha = an
        return

    def predict(self, X):
        y_pred = 0
        print(y_pred)
        return y_pred


if __name__ == '__main__':
    print(123)
    data = make_classification(n_samples=10, n_features=5, n_informative=2, n_redundant=0, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.3)
    svm = SVMachine()
    svm.fit(X_train, y_train)
