import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    # TODO сделать список параметров для каждого датасета
    dataset = datasets.load_iris()
    # dataset = datasets.load_breast_cancer()
    # dataset = datasets.load_digits()
    # dataset = datasets.load_wine()

    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.33, random_state=42)

    # digits
    # model = MLPClassifier(solver='sgd', activation='logistic', alpha=0.05, learning_rate_init=0.01,
    #                       hidden_layer_sizes=(90, 25), max_iter=10000)

    model = MLPClassifier(solver='sgd', activation='logistic', alpha=0.0001, learning_rate_init=0.04,
                          hidden_layer_sizes=(15, 12), max_iter=10000)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(pred)
    print(y_test)
    print(model.loss_)
    print(model.loss_curve_)

    # TODO рисовать красивую кривую
    d = {'x': [i for i, x in enumerate(model.loss_curve_)], 'y': [x for x in model.loss_curve_]}
    dataFrame = pd.DataFrame(d)
    sns.regplot(x="x", y="y", data=dataFrame, scatter_kws={"s": 10}, ci=None, truncate=False, fit_reg=False)
    plt.show()
