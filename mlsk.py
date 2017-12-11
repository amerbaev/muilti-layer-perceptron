import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == "__main__":
    models = [
        {
            "dataset": datasets.load_iris(),
            "params": {
                'alpha': 0.1,
                'max_iter': 10000
            }
        },
        {
            "dataset": datasets.load_breast_cancer(),
            "params": {
                'solver': 'adam',
                'alpha': 1e-5,
                'hidden_layer_sizes': (100, 40),
                'random_state': 1,
                'max_iter': 10000
            }
        },
        {
            "dataset": datasets.load_digits(),
            "params": {
                "solver": 'sgd',
                'activation': 'logistic',
                'alpha': 0.05,
                'learning_rate_init': 0.01,
                'hidden_layer_sizes': (90, 25),
                'max_iter': 10000
            }
        },
        {
            "dataset": datasets.load_wine(),
            "params": {
                "solver": 'lbfgs',
                'alpha': 1e-7,
                'hidden_layer_sizes': (100),
                'random_state': 1,
                'max_iter': 10000
            }
        },
    ]

    for i in models:
        dataset = i['dataset']
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.33,
                                                            random_state=42)

        model = MLPClassifier(**i['params'])

        model.fit(X_train, y_train)

        plot_learning_curve(model, "Learning", X_train, y_train)
        plt.show()

        pred = model.predict(X_test)
        print(pred)
        print(y_test)
        print(model.loss_)

        d = {'x': [i for i, x in enumerate(model.loss_curve_)], 'y': [x for x in model.loss_curve_]}
        dataFrame = pd.DataFrame(d)
        sns.regplot(x="x", y="y", data=dataFrame, scatter_kws={"s": 0}, order=6, ci=None, truncate=True)
        plt.show()
