# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 13:55
# @Author  : Calvin Ren
# @Email   : rqx12138@163.com
# @File    : BPNN.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import scale
from sklearn import datasets


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    :param x: array
    :return: softmax result
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    """
    Compute relu function for each sets of scores in x.
    :param x: array
    :return: relu result
    """
    return np.maximum(0, x)


def d_relu(x):
    """
    Compute derivative of relu function for each sets of scores in x.
    :param x: array
    :return: derivative of relu result
    """
    return np.where(x > 0, 1, 0)


def loss(y_pre, y_true):
    """
    Compute loss function for each sets of scores in x.
    :param y_pre: predict result
    :param y_true: ground truth
    :return: loss result
    """
    m = y_true.shape[1]
    prob = np.multiply(np.log(y_pre), y_true) + np.multiply(np.log(1 - y_pre), 1 - y_true)
    return -np.sum(prob) / m


def Normalize(data):
    """
    Normalize data
    :param data: array
    :return: normalized data
    """
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]


class BP_NN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, lr_decay=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.w1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros(shape=(hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros(shape=(output_size, 1))
        self.a2 = None
        self.z2 = None
        self.a1 = None
        self.z1 = None
        self.loss_record = []
        self.acc_record = []

    def forward(self, x):
        """
        Forward propagation
        :param x: data
        :return: forward result
        """
        self.z1 = np.dot(self.w1, x) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, x, y):
        """
        Backward propagation
        :param x: data
        :param y: ground truth
        :return: None
        """
        delta2 = self.a2 - y
        delta1 = np.multiply(np.dot(self.w2.T, delta2), d_relu(self.a1))
        m = x.shape[1]
        self.w2 -= self.learning_rate * np.dot(delta2, self.a1.T) / m
        self.b2 -= self.learning_rate * np.sum(delta2, axis=1, keepdims=True) / m
        self.w1 -= self.learning_rate * np.dot(delta1, x.T) / m
        self.b1 -= self.learning_rate * np.sum(delta1, axis=1, keepdims=True) / m

    def train(self, x, y, epoch):
        """
        Train model
        :param x: data
        :param y: ground truth
        :param epoch: Train epoch
        :return: None
        """
        for i in range(epoch):
            if i == 5000 and self.lr_decay:
                self.learning_rate = 0.20
            if i == 15000 and self.lr_decay:
                self.learning_rate = 0.1
            if i == 40000 and self.lr_decay:
                self.learning_rate = 0.05
            if i == 30000 and self.lr_decay:
                self.learning_rate = 0.08
            self.forward(x)
            epoch_loss = loss(self.a2, y)
            predict = self.predict(x, y, return_acc=False)
            self.backward(x, y)
            self.loss_record.append(epoch_loss)
            self.acc_record.append(predict)
            if i % 100 == 0:
                print('Epoch:', i, ' || Learning Rate:', self.learning_rate, ' || Loss:', epoch_loss,
                      ' || Accuracy:', predict)
            # if i % 5000 == 0:
            #     if self.lr_decay:
            #         self.learning_rate = self.learning_rate * (self.lr_decay ** (i / 10000))

    def predict(self, x_predict, y_true, return_acc=False):
        """
        Predict result
        :param x_predict: data
        :param y_true: ground truth
        :param return_acc: return accuracy or not
        :return: accuracy
        """
        predict_y = self.forward(x_predict)
        n_row, n_col = predict_y.shape
        res = np.empty(shape=(n_row, n_col), dtype=int)

        for i in range(n_row):
            for j in range(n_col):
                if predict_y[i, j] > 0.5:
                    res[i, j] = 1
                else:
                    res[i, j] = 0
        true_count = 0
        for k in range(y_true.shape[1]):
            data_len = y_true.shape[0]
            classify_check = True
            for i in range(data_len):
                if res[i][k] != y_true[i][k]:
                    classify_check = False
                    break
            if classify_check:
                true_count += 1
            else:
                if return_acc:
                    print('第', k, '个数据分类错误')

        acc = true_count / y_true.shape[1] * 100
        if return_acc:
            print('Accuracy: %.4f%%' % acc)
        return acc


def data_load(dataset):
    """
    Load data
    :param dataset: iris or wine
    :return: data and label
    """
    if dataset == 'iris':
        dataset = datasets.load_iris()
        tmp_data = pd.DataFrame(data=dataset.data)
        tmp_data['target'] = dataset.target
        sb.pairplot(data=tmp_data, diag_kind='hist', hue='target', palette='Paired')
        data = dataset.data
        label = dataset.target
        x = scale(data)
        y = pd.get_dummies(label).values

        return x, y

    elif dataset == 'wine':
        dataset = datasets.load_wine()
        tmp_data = pd.DataFrame(data=dataset.data)
        tmp_data['target'] = dataset.target
        sb.pairplot(data=tmp_data, diag_kind='hist', hue='target', palette='Paired')
        data = dataset.data
        label = dataset.target
        label = pd.get_dummies(label).values
        x = scale(data)
        y = label

        return x, y


if __name__ == '__main__':
    x, y = data_load('iris')  # Load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # Split data
    model = BP_NN(4, 16, 3, 0.1)  # Create BP model
    model.train(x_train.T, y_train.T, 2000)  # Train model
    model.predict(x_test.T, y_test.T, return_acc=True)  # Predict result
    plt.plot(model.loss_record)  # Plot loss
    plt.title('Loss Record')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(model.acc_record)  # Plot accuracy
    plt.title('Accuracy Record')
    plt.xlabel('epoch')
    plt.ylabel('Learning Rate')
    plt.show()

    # wine_x, wine_y = data_load('wine')  # Load data
    # x_train, x_test, y_train, y_test = train_test_split(wine_x, wine_y, test_size=0.2)  # Split data
    # model = BP_NN(13, 32, 3, 0.2, lr_decay=True)  # Create BP model
    # model.train(x_train.T, y_train.T, 500)  # Train model
    # model.predict(x_test.T, y_test.T, return_acc=True)  # Predict result
    # plt.plot(model.loss_record)  # Plot loss
    # plt.title('Loss Record')
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # plt.show()
    #
    # plt.plot(model.acc_record) # Plot accuracy
    # plt.title('Accuracy Record')
    # plt.xlabel('epoch')
    # plt.ylabel('Learning Rate')
    # plt.show()
