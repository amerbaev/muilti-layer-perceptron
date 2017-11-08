import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, preprocessing, metrics
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
            l, v = slice_dataset_2to1(x, y)
            x_t, y_t, x_v, y_v = l['x'], l['y'], v['x'], v['y']
            for _ in tqdm(range(self.num_iter)):
                j = np.random.randint(0, len(x_t))
                a_layer = [np.concatenate((np.array([x_t[j]]).transpose(), [[-1]]))]
                for l in range(self.hidden_layers + 1):
                    z = np.dot(weights[l], a_layer[l])  # 3x1
                    a_res = np.concatenate((V_SIGMOID(z), [[-1]]))
                    a_layer.append(a_res)

                error = np.array(y_t[j]) - a_layer[-1][:-1:]  # 1x1
                weights[-1] = weights[-1] + np.dot(error, a_layer[-2].transpose()) * rate
                for k in range(len(weights) - 1, 0, -1):
                    error = (np.dot(weights[k].transpose(), error) * a_layer[k] * (1 - a_layer[k]))[:-1:]
                    weights[k - 1] = weights[k - 1] + np.dot(error, a_layer[k - 1].transpose()) * rate

                # predict = self.__predict_test(x_v, weights)
                # logloss = self.logloss_crutch(y_v, predict)
                # if rate in self.errors:
                #     self.errors[rate].append(logloss)
                # else:
                #     self.errors[rate] = [logloss]

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
            f, axarr = plt.subplots(len(self.errors))
            for i, k in enumerate(self.errors):
                axarr[i].plot([e for e in self.errors[k]], alpha=.33)
                axarr[i].set_title('Rate ' + str(k))
        else:
            key = next(iter(self.errors))
            plt.plot([e for e in self.errors[key]], alpha=.33)

    def plot_losses(self):
        plt.plot(self.losses, label='Logloss')
        plt.xticks(range(len(self.rates)), self.rates)
        plt.legend()


def plot_confusion_matrix(y_pred, y_true, classes, title='Матрица ошибок'):
    cm = metrics.confusion_matrix(y_true, y_pred)

    plt.figure()
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


def plot_multiclass_roc(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    n_classes = len(y_true[0])

    fpr = [None] * n_classes
    tpr = [None] * n_classes
    roc_auc = [None] * n_classes
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    plt.figure()
    plt.plot((0, 1), (0, 1), linestyle='--')
    for i, (f, t, a) in enumerate(zip(fpr, tpr, roc_auc)):
        plt.plot(f, t, label='ROC класс {} (площадь = {:.2f})'.format(i, a))
    plt.xlabel('False-positive')
    plt.ylabel('True-positive')
    plt.legend()


def load_dataset(dataset, perceptron):
    X = preprocessing.normalize(dataset.data, norm='l2')
    lb = preprocessing.LabelBinarizer()
    Y = [np.array([i]).transpose() for i in lb.fit_transform(dataset.target)]

    l, v = slice_dataset_2to1(X, Y)
    perceptron.train(l['x'], l['y'])
    # perceptron.plot_losses()
    # perceptron.plot_errors()
    result = perceptron.predict(v['x'])
    # print(result)
    # test = perceptron.logloss_crutch(v['y'], result)

    pred = lb.inverse_transform(np.array(result))
    true = lb.inverse_transform(v['y']).transpose()[0]
    plot_multiclass_roc(result, lb.fit_transform(true))
    # plot_confusion_matrix(y_pred=pred, y_true=true, classes=dataset.target_names)

    print(pred)
    print(true)
    plt.show()


if __name__ == "__main__":
    data = datasets.load_iris()
    # data = datasets.load_digits()
    # data = datasets.load_breast_cancer()
    # data = datasets.load_wine()
    a = Perceptron(4, 3, (7,), [0.05], 50000)  # iris success
    # a = Perceptron(64, 10, (150,), [0.01, 0.005, 0.001, 0.1], 30000)  # digits success
    # a = Perceptron(30, 2, (100,), [0.01, 0.005, 0.001, 0.1], 100)  # cancer
    # a = Perceptron(13, 3, (15,), [0.01, 0.001, 0.1], 10000)  # wine

    load_dataset(data, a)
