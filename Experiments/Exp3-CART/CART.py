# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 00:38
# @Author  : Calvin Ren
# @Email   : rqx12138@163.com
# @File    : CART.py
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# Classification and Regression Tree
class CART(object):
    def __init__(self, tree='cls', criterion='gini', prune='depth', max_depth=4, min_criterion=0.05):
        self.feature = None
        self.label = None
        self.n_samples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.depth = 0
        self.root = None
        self.criterion = criterion
        self.prune = prune
        self.max_depth = max_depth
        self.min_criterion = min_criterion
        self.tree = tree

    # 模型拟合
    def fit(self, features, target):
        self.root = CART()
        if (self.tree == 'cls'):
            self.root._grow_tree(features, target, self.criterion)
        else:
            self.root._grow_tree(features, target, 'mse')
        self.root._prune(self.prune, self.max_depth, self.min_criterion, self.root.n_samples)

    # 模型预测
    def predict(self, features):
        return np.array([self.root._predict(f) for f in features])

    # 决策树可视化
    def print_tree(self):
        self.root._show_tree(0, ' ')

    # 决策树生成
    def _grow_tree(self, features, target, criterion='gini'):
        self.n_samples = features.shape[0]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        if criterion in {'gini', 'entropy'}:
            self.label = max([(c, len(target[target == c])) for c in np.unique(target)], key=lambda x: x[1])[0]
        else:
            self.label = np.mean(target)

        impurity_node = self._calc_impurity(criterion, target)

        for col in range(features.shape[1]):
            feature_level = np.unique(features[:, col])
            thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

            for threshold in thresholds:
                target_l = target[features[:, col] <= threshold]
                impurity_l = self._calc_impurity(criterion, target_l)
                n_l = float(target_l.shape[0]) / self.n_samples

                target_r = target[features[:, col] > threshold]
                impurity_r = self._calc_impurity(criterion, target_r)
                n_r = float(target_r.shape[0]) / self.n_samples

                impurity_gain = impurity_node - (n_l * impurity_l + n_r * impurity_r)
                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = col
                    best_threshold = threshold

        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        self._split_tree(features, target, criterion)

    def _split_tree(self, features, target, criterion):
        features_l = features[features[:, self.feature] <= self.threshold]
        target_l = target[features[:, self.feature] <= self.threshold]
        self.left = CART()
        self.left.depth = self.depth + 1
        self.left._grow_tree(features_l, target_l, criterion)

        features_r = features[features[:, self.feature] > self.threshold]
        target_r = target[features[:, self.feature] > self.threshold]
        self.right = CART()
        self.right.depth = self.depth + 1
        self.right._grow_tree(features_r, target_r, criterion)

    # 计算
    def _calc_impurity(self, criterion, target):
        if criterion == 'gini':
            return 1.0 - sum(
                [(float(len(target[target == c])) / float(target.shape[0])) ** 2.0 for c in np.unique(target)])
        elif criterion == 'mse':
            return np.mean((target - np.mean(target)) ** 2.0)
        else:
            entropy = 0.0
            for c in np.unique(target):
                p = float(len(target[target == c])) / target.shape[0]
                if p > 0.0:
                    entropy -= p * np.log2(p)
            return entropy

    # 剪枝
    def _prune(self, method, max_depth, min_criterion, n_samples):
        if self.feature is None:
            return

        self.left._prune(method, max_depth, min_criterion, n_samples)
        self.right._prune(method, max_depth, min_criterion, n_samples)

        pruning = False

        if method == 'impurity' and self.left.feature is None and self.right.feature is None:
            if (self.gain * float(self.n_samples) / n_samples) < min_criterion:
                pruning = True
        # 深度
        elif method == 'depth' and self.depth >= max_depth:
            pruning = True

        if pruning is True:
            self.left = None
            self.right = None
            self.feature = None

    def _predict(self, d):
        if self.feature != None:
            if d[self.feature] <= self.threshold:
                return self.left._predict(d)
            else:
                return self.right._predict(d)
        else:
            return self.label

    # 决策树可视化
    def _show_tree(self, depth, cond):
        base = '    ' * depth + cond
        if self.feature != None:
            print(base + 'if X[' + str(self.feature) + '] <= ' + str(self.threshold))
            self.left._show_tree(depth + 1, 'then ')
            self.right._show_tree(depth + 1, 'else ')
        else:
            print(base + '{value: ' + str(self.label) + ', samples: ' + str(self.n_samples) + '}')


def vis_dataset(dataset):
    tmp_data = pd.DataFrame(data=dataset.data)
    tmp_data['target'] = dataset.target
    sb.pairplot(data=tmp_data, diag_kind='hist', hue='target', palette='Paired')


# 生成分类决策树
def classification(dataset):
    print('Classification Tree')
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    cls = CART(tree='cls', criterion='gini', prune='depth', max_depth=3)
    cls.fit(X_train, y_train)
    # cls.print_tree()

    pred = cls.predict(X_test)
    print("Predict Result: ", pred[:10])
    print("Ground Truth: ", y_test[:10])
    print("Prediction Accuracy:    {}".format(sum(pred == y_test) / len(pred)))


# 生成回归决策树
def regression(dataset):
    print('Regression Tree')
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Fit regression model
    reg = CART(tree='reg', criterion='mse', prune='depth', max_depth=2)
    reg.fit(X_train, y_train)
    # reg.print_tree()

    pred = reg.predict(X_test)
    print("Predict Result: ", pred[:8])
    print("Ground Truth: ", y_test[:8])
    cart_loss = abs((np.sum(pred) - np.sum(y_test)) / len(y_test))
    print('CART Tree Loss is : ', cart_loss)


if __name__ == "__main__":
    iris = load_iris()
    vis_dataset(iris)

    wine = load_wine()
    vis_dataset(wine)

    classification(iris)
    classification(wine)

    diabetes = load_diabetes()
    cancer = load_breast_cancer()

    regression(diabetes)
    regression(cancer)