import math

import tensorflow as tf


# Для начала посмотрите ниже класс MLP
# Один слой перцептрона
class Layer(object):
    # input_sizes - массив количеств входов, почему массив см конструктор MLP ниже
    # output_size - количество выходов
    # scope - строчка, переменные в TensorFlow можно организовывать в скоупы
    #   для удобного переиспользования (см https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html)
    def __init__(self, input_sizes, output_size, scope):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.scope = scope or "Layer"

        # входим в скоуп
        with tf.variable_scope(self.scope):
            # массив нейронов
            self.Ws = []
            for input_idx, input_size in enumerate(input_sizes):
                # идентификатор нейрона
                W_name = "W_%d" % (input_idx,)
                # инициализатор весов нейрона - равномерное распределение
                W_initializer = tf.random_uniform_initializer(
                    -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
                # создание нейрона - как матрицы input_size x output_size
                W_var = tf.get_variable(W_name, (input_size, output_size), initializer=W_initializer)
                self.Ws.append(W_var)
            # создание вектора свободных членов слоя
            # этот вектор будет прибавлен к выходам нейронов слоя
            self.b = tf.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

    # использование слоя нейронов
    # xs - вектор входных значений
    # возвращает вектор выходных значений
    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws), \
            "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
        with tf.variable_scope(self.scope):
            # рассчет выходных значений
            # так как каждый нейрон - матрица
            # то вектор выходных значений - это сумма
            # умножений матриц-нейронов на входной вектор + вектор свободных членов
            return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

    # возвращает список параметров слоя
    # это нужно для работы алгоритма обратного распространения ошибки
    def variables(self):
        return [self.b] + self.Ws

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                                initializer=lambda x, dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)


# Многослойный перцептрон
class MLP(object):
    # input_sizes - массив размеров входных слоев, не знаю зачем,
    #   но здесь реализована поддержка нескольких входных слоев,
    #   выглядит это как один входной слой разделенный на части,
    #   по факту эта возможность не используется, то есть входной слой один
    # hiddens - массив размеров скрытых слоев, по факту
    #   используется 2 скрытых слоя по 100 нейронов
    #   и выходной слой - 4 нейрона, что интересно, не делать ничего
    #   нейросеть не может, такого варианта у нее нет
    # nonlinearities - массив передаточных функций нейронов слоев, про передаточные функции см <a href="https://ru.wikipedia.org/wiki/%D0%98%D1%81%D0%BA%D1%83%D1%81%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD">Искусственный нейрон</a>
    # scope - строчка, переменные в TensorFlow можно организовывать в скоупы
    #   для удобного переиспользования
    # given_layers - можно передать уже созданные слои
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"

        assert len(hiddens) == len(nonlinearities), \
            "Number of hiddens must be equal to number of nonlinearities"

        with tf.variable_scope(self.scope):
            if given_layers is not None:
                # использовать переданные слои
                self.input_layer = given_layers[0]
                self.layers = given_layers[1:]
            else:
                # создать слои
                # создание входного слоя
                self.input_layer = Layer(input_sizes, hiddens[0], scope="input_layer")
                self.layers = []
                # создать скрытые слои
                for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
                    self.layers.append(Layer(h_from, h_to, scope="hidden_layer_%d" % (l_idx,)))

    # использование нейросети
    # xs - вектор входных значений
    # возвращается выход выходного слоя
    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        with tf.variable_scope(self.scope):
            # применение входного слоя к вектору входных значений
            hidden = self.input_nonlinearity(self.input_layer(xs))
            for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
                # применение скрытых слоев в выходам предидущих слоев
                hidden = nonlinearity(layer(hidden))
            return hidden

    # список параметров всей нейронной сети от входного слоя к выходному
    def variables(self):
        res = self.input_layer.variables()
        for layer in self.layers:
            res.extend(layer.variables())
        return res

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        nonlinearities = [self.input_nonlinearity] + self.layer_nonlinearities
        given_layers = [self.input_layer.copy()] + [layer.copy() for layer in self.layers]
        return MLP(self.input_sizes, self.hiddens, nonlinearities, scope=scope,
                   given_layers=given_layers)

