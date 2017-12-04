import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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
        pred = model.predict(X_test)
        print(pred)
        print(y_test)
        print(model.loss_)


        # TODO рисовать красивую кривую
        d = {'x': [i for i, x in enumerate(model.loss_curve_)], 'y': [x for x in model.loss_curve_]}
        dataFrame = pd.DataFrame(d)
        sns.regplot(x="x", y="y", data=dataFrame, scatter_kws={"s": 10}, ci=None, truncate=False, fit_reg=False)
        plt.show()
