# TODO добавить регуляризатор
# TODO разбить датасет на обучающую, валидационную и тестовую выборки и использовать их при обучении нейросетки

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import copy

V_SIGMOID = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))

iris = datasets.load_iris()
IRIS_X = iris.data
mapIris = {
    0: [[1], [0], [0]],
    1: [[0], [1], [0]],
    2: [[0], [0], [1]],
}
IRIS_Y = np.array([mapIris[x] for x in iris.target])

# LEARN_X = [
#     (0, 0),
#     (0, 1),
#     (1, 0),
#     (1, 1)
# ]
#
# LEARN_Y = (
#     0,
#     1,
#     1,
#     0
# )
LEARN_X = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]

LEARN_Y = (
    0,
    1,
    1,
    0,
    1,
    0,
    0,
    1
)


def slice_dataset_2to1(dataset):
    np.random.shuffle(dataset)
    edge = int(2 / 3 * len(dataset))
    x_learn, x_validation = dataset[:edge], dataset[edge:]
    return x_learn, x_validation


class Perceptron:
    logloss = None

    def __init__(self, n_inputs: int, outputs: int, neurons: tuple, rates: list, num_iter: int):
        self.hidden_layers = len(neurons)
        self.error = 1000000
        self.rates = rates
        self.num_iter = num_iter
        self.weights = [np.random.rand(neurons[0], n_inputs)]
        for i in range(len(neurons) - 1):
            self.weights.append(np.random.rand(neurons[i + 1], neurons[i]))
        self.weights.append(np.random.rand(outputs, neurons[-1]))

    def train(self, x, y):
        losses = []
        init_weights = []
        for i in range(len(self.weights)):
            init_weights.append(np.random.rand(self.weights[i].shape[0], self.weights[i].shape[1]))
        for rate in self.rates:
            print('Rate: ', rate)
            errors = []
            weights = copy.deepcopy(init_weights)
            for i in range(self.num_iter):
                for j in range(len(x)):
                    a_layer = [np.array([x[j]]).transpose()]
                    for l in range(self.hidden_layers + 1):
                        z = np.dot(weights[l], a_layer[l])  # 3x1
                        a_layer.append(V_SIGMOID(z))

                    error = np.array(y[j]) - a_layer[-1]  # 1x1
                    errors.append(error[0])
                    weights[-1] = weights[-1] + np.dot(error, a_layer[-2].transpose()) * rate

                    for k in range(len(weights) - 1, 0, -1):
                        error = np.dot(weights[k].transpose(), error) * a_layer[k] * (1 - a_layer[k])
                        weights[k - 1] = weights[k - 1] + np.dot(error, a_layer[k - 1].transpose()) * rate

            predict = self.__predict_test(x, weights)
            logloss = self.logloss_crutch(y, predict)
            losses.append(logloss)
            if not self.logloss or logloss < self.logloss:
                self.logloss = logloss
                self.weights = weights
                print('Better logloss:', logloss)
            else:
                print('Worst logloss: ', logloss)

        plt.plot(losses)
        plt.show()

    def predict(self, x):
        arr = []
        for n in range(len(x)):
            a_l = np.array([x[n]]).transpose()  # 3x1
            for i in range(len(self.weights)):
                z = np.dot(self.weights[i], a_l)  # 3x1
                a_l = V_SIGMOID(z)  # 3x1

            arr.append(a_l)
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
            a_l = np.array([x[n]]).transpose()  # 3x1
            for i in range(len(weights)):
                z = np.dot(weights[i], a_l)  # 3x1
                a_l = V_SIGMOID(z)  # 3x1

            arr.append(a_l)
        return arr


a = Perceptron(4, 3, (15,), [1, 0.1, 0.05, 0.01, 0.005, 0.001], 1000)
iris_l, iris_v = slice_dataset_2to1(IRIS_X)
a.train(iris_l, IRIS_Y)
result = a.predict(iris_v)
test = a.logloss_crutch(IRIS_Y, result)
print('\n', test)
