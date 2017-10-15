# TODO добавить регуляризатор
# TODO разбить датасет на обучающую, валидационную и тестовую выборки и использовать их при обучении нейросетки
# TODO при обучении использовать нормальную функцию оценки ошибки
# https://habrahabr.ru/company/ods/blog/328372/ (ниже указал где надо поменять)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

V_SIGMOID = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))

iris = datasets.load_iris()
IRIS_X = iris.data
mapIris = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1],
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


class Perceptron:
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
        for rate in self.rates:
            errors = []
            weights = []
            for i in range(len(self.weights)):
                weights.append(np.random.rand(self.weights[i].shape[0], self.weights[i].shape[1]))

            for i in range(self.num_iter):
                for j in range(len(x)):
                    a_layer = [np.array([x[j]]).transpose()]
                    for l in range(self.hidden_layers + 1):
                        z = np.dot(weights[l], a_layer[l])  # 3x1
                        a_layer.append(V_SIGMOID(z))

                    error = np.array([y[j]]).transpose() - a_layer[-1]  # 1x1
                    errors.append(error[0])
                    weights[-1] = weights[-1] + np.dot(error, a_layer[-2].transpose()) * rate

                    for k in range(len(weights) - 1, 0, -1):
                        error = np.dot(weights[k].transpose(), error) * a_layer[k] * (1 - a_layer[k])
                        weights[k - 1] = weights[k - 1] + np.dot(error, a_layer[k - 1].transpose()) * rate

            # вот эту херь ниже надо поменять.
            # Первое: она не работает если есть несколько выходных классов
            # Второе она смотрит ошибку на последнем элементе обучающей выборки, что вообще не правильно.
            # Как должно быть:
            # Мы, после того как обучили, делаем предсказание на этой же выборке и смотрим по какой либо метрике,
            # которая указана в статье на хабре (ссылка выше)
            # И только после этого проверяем метрика улучшилась или нет
            if abs(errors[-1][0]) < self.error:
                for i in range(len(self.weights)):
                    self.weights[i] = weights[i]
                self.error = abs(errors[-1][0])
            # plt.plot(errors)
            # plt.show()

    def predict(self, x):
        arr = []
        for n in range(len(x)):
            a_l = np.array([x[n]]).transpose()  # 3x1
            for i in range(len(self.weights)):
                z = np.dot(self.weights[i], a_l)  # 3x1
                a_l = V_SIGMOID(z)  # 3x1

            arr.append(a_l)
        return np.array(arr)  # TODO: получается shape (150, 3, 1)

    @staticmethod
    def logloss_crutch(y_true, y_pred):
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


a = Perceptron(4, 3, (15,), [0.01], 1000)
a.train(IRIS_X, IRIS_Y)
result = a.predict(IRIS_X)
print(result.shape)
print(IRIS_Y.shape)
# test = a.logloss_crutch(IRIS_Y, result)
# print(test)
