import copy
import itertools

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics

import matplotlib.pyplot as plt

from tqdm import tqdm

V_SIGMOID = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))


def slice_dataset_2to1(conditions, results):
    dataset = [(x, y) for x, y in zip(conditions, results)]
    np.random.shuffle(dataset)
    edge = int(2 / 3 * len(dataset))
    learn, validation = dataset[:edge], dataset[edge:]

    learn = {
        'x': np.array([d[0] for d in learn]),
        'y': np.array([d[1] for d in learn])
    }

    validation = {
        'x': np.array([d[0] for d in validation]),
        'y': np.array([d[1] for d in validation])
    }

    return learn, validation


class Perceptron:
    logloss = None

    def __init__(self, n_inputs: int, outputs: int, neurons: tuple, rates: list, num_iter: int):
        self.hidden_layers = len(neurons)
        self.error = 1000000
        self.rates = rates
        self.num_iter = num_iter
        if len(neurons) > 0:
            self.weights = [np.random.rand(neurons[0], n_inputs + 1)]
            for i in range(len(neurons) - 1):
                self.weights.append(np.random.rand(neurons[i + 1], neurons[i] + 1))
            self.weights.append(np.random.rand(outputs, neurons[-1] + 1))
        else:
            self.weights = [np.random.rand(outputs, n_inputs + 1)]
        self.errors = {}
        self.losses = []

    def train(self, x, y):
        init_weights = []
        for i in range(len(self.weights)):
            init_weights.append(np.random.rand(self.weights[i].shape[0], self.weights[i].shape[1]))
        for rate in self.rates:
            print('\nRate: ', rate)
            weights = copy.deepcopy(init_weights)
            for i in tqdm(range(self.num_iter)):
                for j in range(len(x)):
                    a_layer = [np.concatenate((np.array([x[j]]).transpose(), [[-1]]))]
                    for l in range(self.hidden_layers + 1):
                        z = np.dot(weights[l], a_layer[l])  # 3x1
                        a_layer.append(np.concatenate((V_SIGMOID(z), [[-1]])))

                    error = np.array(y[j]) - a_layer[-1][:-1:]  # 1x1
                    if rate in self.errors:
                        self.errors[rate] += [e for e in error]
                    else:
                        self.errors[rate] = [e for e in error]

                    weights[-1] = weights[-1] + np.dot(error, a_layer[-2].transpose()) * rate
                    for k in range(len(weights) - 1, 0, -1):
                        error = (np.dot(weights[k].transpose(), error) * a_layer[k] * (1 - a_layer[k]))[:-1:]
                        weights[k - 1] = weights[k - 1] + np.dot(error, a_layer[k - 1].transpose()) * rate

            predict = self.__predict_test(x, weights)
            logloss = self.logloss_crutch(y, predict)
            self.losses.append(logloss)
            if not self.logloss or logloss < self.logloss:
                self.logloss = logloss
                self.weights = weights
                print('Better logloss:', logloss)
            else:
                print('Worst logloss: ', logloss)

    def predict(self, x):
        arr = []
        for n in range(len(x)):
            a_l = np.concatenate((np.array([x[n]]).transpose(), [[-1]]))  # 3x1
            for i in range(len(self.weights)):
                z = np.dot(self.weights[i], a_l)  # 3x1
                a_l = np.concatenate((V_SIGMOID(z), [[-1]]))  # 3x1

            arr.append(a_l[:-1:].transpose()[0])
        return arr

    def logloss_crutch(self, y_true, y_pred):
        logloss_sum = self.__logloss_sum(y_true, y_pred)
        logloss = - (1 / y_true.size) * logloss_sum
        return logloss

    def __logloss_sum(self, y_true, y_pred):
        try:
            logloss_sum = sum(self.__logloss_sum(true_elem, pred_elem) for true_elem, pred_elem in zip(y_true, y_pred))
        except TypeError:
            logloss_sum = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        return logloss_sum

    @staticmethod
    def __predict_test(x, weights):
        arr = []
        for n in range(len(x)):
            a_l = np.concatenate((np.array([x[n]]).transpose(), [[-1]]))  # 3x1
            for i in range(len(weights)):
                z = np.dot(weights[i], a_l)  # 3x1
                a_l = np.concatenate((V_SIGMOID(z), [[-1]]))  # 3x1

            arr.append(a_l[:-1:])
        return arr

    def plot_errors(self):
        f, axarr = plt.subplots(len(self.errors))
        if len(self.errors) > 1:
            for i, k in enumerate(self.errors):
                axarr[i].plot([e for e in self.errors[k]])
                axarr[i].set_title('Rate ' + str(k))
        else:
            axarr.plot([e for e in self.errors[next(iter(self.errors))]])

    def plot_losses(self):
        plt.plot(self.losses, label='Logloss')
        plt.legend()


def plot_confusion_matrix(y_pred, y_true, classes, title='Матрица ошибок'):
    cm = metrics.confusion_matrix(y_true, y_pred)

    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')


if __name__ == "__main__":
    iris = datasets.load_iris()
    IRIS_X = iris.data

    lb = preprocessing.LabelBinarizer()
    IRIS_Y = [np.array([i]).transpose() for i in lb.fit_transform(iris.target)]

    a = Perceptron(4, 3, (4, 3), [0.01, 0.005], 1000)
    iris_l, iris_v = slice_dataset_2to1(IRIS_X, IRIS_Y)
    a.train(iris_l['x'], iris_l['y'])
    # a.plot_losses()
    a.plot_errors()
    # result = a.predict(iris_v['x'])
    # print(result)
    # test = a.logloss_crutch(iris_v['y'], result)
    #
    # pred = lb.inverse_transform(np.array(result))
    # true = lb.inverse_transform(iris_v['y']).transpose()[0]
    # plot_confusion_matrix(y_pred=pred, y_true=true, classes=iris.target_names)

    plt.show()
