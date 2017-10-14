# TODO добавить регуляризатор, обобщить на любое количество слоев, использовать датасет с ирисами
# from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

sigmoid = lambda x: 1 / (1 + np.exp(-x))
v_sigmoid = np.vectorize(sigmoid)

# iris = datasets.load_iris()
# IRIS_X = iris.data[:, :2]  # we only take the first two features.
# IRIS_Y = iris.target


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

N_INPUTS = len(LEARN_X[0])
N_LAYERS = 1
N_NEURONS = (10,)
N_OUTPUTS = 1

GLOBAL_W_1 = np.random.rand(N_NEURONS[0], N_INPUTS)
# GLOBAL_W_2 = np.random.rand(N_NEURONS[1], N_NEURONS[0])
GLOBAL_W_2 = np.random.rand(N_OUTPUTS, N_NEURONS[-1])
RATES = [0.1]
N = 200
GLOBAL_ERROR = 1000000

for RATE in RATES:
    errors = []
    W_1 = np.random.rand(N_NEURONS[0], N_INPUTS)  # 3x4
    # W_2 = np.random.rand(N_NEURONS[1], N_NEURONS[0])
    W_2 = np.random.rand(N_OUTPUTS, N_NEURONS[-1])
    print(W_1, W_2, sep='\n')
    error = 0
    for i in range(N):
        for n in range(len(LEARN_X)):
            A_1 = np.array([LEARN_X[n]]).transpose()  # 4x1
            Z_2 = np.dot(W_1, A_1)  # 3x1
            A_2 = v_sigmoid(Z_2)  # 4x1
            Z_3 = np.dot(W_2, A_2)  # 1x1
            A_R = v_sigmoid(Z_3)  # 1x1

            error = LEARN_Y[n] - A_R  # 1x1
            W_2 = W_2 + np.dot(error, A_2.transpose()) * RATE  # 1x4

            error_1 = np.dot(W_2.transpose(), error) * A_2 * (1 - A_2)  # 3x1
            W_1 = W_1 + np.dot(error_1, A_1.transpose()) * RATE  # 3x2
        errors.append(error[0])
        # if abs(error[0]) < RATE / 2:
        #     break
    if abs(error[0]) < GLOBAL_ERROR:
        GLOBAL_W_1 = W_1
        GLOBAL_W_2 = W_2
        GLOBAL_ERROR = abs(error[0])
    plt.plot(errors)
    plt.show()

for n in range(len(LEARN_X)):
    A_1 = np.array([LEARN_X[n]]).transpose()  # 3x1
    Z_2 = np.dot(GLOBAL_W_1, A_1)  # 3x1
    A_2 = v_sigmoid(Z_2)  # 3x1
    Z_3 = np.dot(GLOBAL_W_2, A_2)  # 1x1
    A_R = v_sigmoid(Z_3)  # 1x1

    print(LEARN_Y[n], A_R)
