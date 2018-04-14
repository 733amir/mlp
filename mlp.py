from activation_functions import ActivationFunctions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from random import shuffle


class MLP:
    """
    This is a Multi-Layer Perceptron implementation. This implementation allows you to have any number of input neurons
    in input layer, any number of hidden layers each including equal number of neurons and any number of outputs. Each
    neurons output activation functions is a Sigmoid function except output (last) layer that use Softmax functions as
    activation function to output probability for different classes (targets).
    """

    _debug = True
    _verbosity = 0
    _min = {
        'in': 1,
        'out': 1,
        'hid_layer': 0,
        'hid_neuron': 1,
    }
    _normal_parameters = [0, 1]

    def __init__(self, input_layer, hidden_layers, hidden_neurons_per_layer, output_layer, random_state=1):
        f"""
        Initialize our Multi-Layer Perceptron model.

        :param input_layer: Number of input neurons in first layer. [possible minimum is {self._min['in']}]
        :param hidden_layers: Number of hidden layers. [possible minimum is {self._min['hid_layer']}]
        :param hidden_neurons_per_layer: Number of hidden neurons per layer.
            [possible minimum is {self._min['hid_neuron']}]
        :param output_layer: Number of output neurons in last layer. [possible minimum is {self._min['out']}]
        :param random_state: Random state for internal random number generator. Using the same number we
            will be able to regenerate the results.
        """

        # Input evaluations.
        if input_layer < 1 or hidden_layers < 0 or \
                (hidden_layers > 0 and hidden_neurons_per_layer < 1) or output_layer < 1:
            raise ValueError(f'Number of neurons and layers in not valid.\n'
                             f'\tInput Layer Neurons [minimum is {self._min["in"]}]: {input_layer}\n'
                             f'\tHidden Layers [minimum is {self._min["hid_layer"]}]: {hidden_layers}\n'
                             f'\tHidden Neurons [minimum is {self._min["hid_neuron"]}]: {hidden_neurons_per_layer}\n'
                             f'\tOutput Layer Neurons [minimum is {self._min["out"]}]: {output_layer}')

        self._nn_dimensions = input_layer, hidden_layers, hidden_neurons_per_layer, output_layer

        # TODO get activation functions from user.
        self._nn_activation_funcs = ActivationFunctions.init_linear_sigmoids_softmax(hidden_layers)

        # Initializing NumPy random number generator.
        self._random_state = random_state
        self._random_generator = np.random.RandomState(self._random_state)

        # Initializing weights and biases of our model using Normal distribution with mean and variance configurable.
        self._weights = self._initialize_weights(*self._nn_dimensions)
        self._biases = self._initialize_biases(*self._nn_dimensions)

        self._log('Initialized weights: [layer][neuron, weight]', self._weights, verbosity=2)
        self._log('Initialized biases: [layer][bias]', self._biases, verbosity=2)

        # Used variables during training.
        self._inputs, self._outputs = [], []
        self._eta, self._errors = 0, []

    def _initialize_weights(self, i, hl, hn, o):  # Input, Hidden layers, Hidden neurons, Output sizes.
        """
        Initialize weights of our Neural Network.

        :param i: Number of neurons in input layer.
        :param hl: Number of hidden layers.
        :param hn: Number of neurons in each hidden layer.
        :param o: Number of neurons in output layer (Number of categories of target).
        :return: Python List of NumPy Arrays that represent weights of our Neural Network.
        """
        sizes = [(i, o)] if hl == 0 else ([(i, hn)] + [(hn, hn)] * (hl - 1) + [(hn, o)])
        return [np.eye(i)] + [self._normal_random(s) for s in sizes]

    def _initialize_biases(self, i, hl, hn, o):  # Input, Hidden layers, Hidden neurons, Output sizes.
        """
        Initialize biases of our Neural Network.

        :param i: Number of neurons in input layer.
        :param hl: Number of hidden layers.
        :param hn: Number of neurons in each hidden layer.
        :param o: Number of neurons in output layer (Number of categories of target).
        :return: Python List of NumPy Arrays that represent biases of our Neural Network.
        """
        sizes = [o] if hl == 0 else [hn] * hl + [o]
        return [np.zeros(i)] + [self._normal_random(s) for s in sizes]

    @staticmethod
    def _initialize_delta(i, hl, hn, o):  # Input, Hidden layers, Hidden neurons, Output sizes.
        """
        Initialize delta for update procedure.

        :param i: Number of neurons in input layer.
        :param hl: Number of hidden layers.
        :param hn: Number of neurons in each hidden layer.
        :param o: Number of neurons in output layer (Number of categories of target).
        :return: Python List of NumPy Arrays that represent delta for our Neural Network.
        """
        sizes = [i] + [hn] * hl + [o]
        return [np.zeros(s) for s in sizes]

    def predict(self, X):
        """
        Calculate output using weights, biases and activation functions of the model.

        :param X: Is a NumPy array representing each observation in a row. It can contains multiple observations.
        :return: A NumPy array containing the result of the observation. Number of columns is equal to output (last)
            layer and number of rows is equal to number of observations (rows) in input data.
        """
        self._log(f'Input for prediction to Network:', [X], verbosity=2)

        self._inputs, self._outputs = [], []
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            self._inputs.append(X.dot(w) + b)
            X = self._nn_activation_funcs[i](self._inputs[-1])
            self._outputs.append(X)

            # Log each batch of calculations.
            self._log(f'Layer #{i + 1} weights (column => neuron):', [w], verbosity=3)
            self._log(f'Layer #{i + 1} bias (column => neuron):', [b], verbosity=3)
            self._log(f'Layer #{i + 1} net input:', [self._inputs[-1]], verbosity=3)
            self._log(f'Layer #{i + 1} output ({self._nn_activation_funcs.get_name(i)}):', [X], verbosity=3)

        self._log(f'Output of Network prediction:', [X], verbosity=2)

        return X

    def fit(self, X, y, epoch=10000, eta=1, batch_size=1000, kfold=10):
        # TODO implement kfold 10
        self._errors = []
        for i in range(epoch):
            for j in range(0, X.shape[0], batch_size):
                start, end = j, min(j + batch_size, X.shape[0])
                batch_X, batch_y = X[start:end], y[start:end]
                self._errors.append(self._batch_update(batch_X, batch_y, eta))

    def _batch_update(self, X, y, eta):
        self.predict(X)
        self._delta = self._initialize_delta(*self._nn_dimensions)

        dmse = y - self._outputs[-1]
        mse = (dmse ** 2 / 2).sum()
        self._log('Mean Squared Error:', [mse], verbosity=1)

        self._log('Weights', [w.shape for w in self._weights], verbosity=4)
        self._log('Delta', [d.shape for d in self._delta], verbosity=4)
        self._log('Output', [o.shape for o in self._outputs], verbosity=4)
        self._log('d MSE', [dmse.shape], verbosity=4)

        self._delta[-1] = dmse * (self._outputs[-1] * (1 - self._outputs[-1]))  # TODO Generalize

        for i in range(-2, -len(self._delta) - 1, -1):
            self._delta[i] = self._delta[i + 1].dot(self._weights[i + 1].T) * \
                             (self._outputs[i] * (1 - self._outputs[i]))  # TODO Generalize

        self._log('Delta matrix:', self._delta, verbosity=2)

        # Updating Parameters
        for i in range(-1, -len(self._delta), -1):
            self._weights[i] += (eta / self._delta[i].shape[0]) * self._outputs[i - 1].T.dot(self._delta[i])
            self._biases[i] += (eta / self._delta[i].shape[0]) * self._delta[i].sum(axis=0)

        self._log('Weights after update:', self._weights, verbosity=3)
        self._log('Biases after update:', self._biases, verbosity=3)

        return mse

    def save_to_file(self):
        pass

    def load_from_file(self):
        pass

    # def _activation_func(self, layer, net_input, raw_input=False):  # TODO
    #     """
    #     Activation function for each of out layers. Layers start at 0 for input layer, 1 for first hidden layer, ... .
    #
    #     :param layer: Number of the layer.
    #     :param net_input: Network input of the layer.
    #     :return: A tuple containing (Applied activation function on net input, Activation function name).
    #     """
    #
    #     if raw_input:
    #         result, activation_func_name = net_input, 'No'
    #
    #     # With Sigmoid for hidden layer(s) and Softmax for output layer.
    #     elif layer == self._get_number_of_layers() - 1:  # If we are at last (output) layer.
    #         result, activation_func_name = self._softmax(net_input), 'Softmax'
    #     else:
    #         result, activation_func_name = self._sigmoid(net_input), 'Sigmoid'
    #
    #     return result, activation_func_name

    # def _get_number_of_layers(self):  # TODO
    #     """
    #     Number of layers including an input and output layer and 0 or more hidden layers.
    #
    #     :return: Number of all layers including input, hidden(s) and outputs.
    #     """
    #
    #     return 1 + self._nn_dimens[1][0] + 1

    def _log(self, title, data, verbosity=1):  # TODO
        if self._debug and self._verbosity >= verbosity:
            print(title, *data, sep='\n', end='\n\n')

    def _normal_random(self, size):  # TODO
        mean, var = self._normal_parameters
        return self._random_generator.normal(loc=mean, scale=var, size=size)


def main():
    ##### Iris dataset information #####
    # Data are ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # Targets are ['setosa', 'versicolor', 'virginica']
    # Number of samples: 150
    # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

    iris = load_iris()
    X, y = iris['data'], iris['target']

    # Converting target to one-hot target.
    mapper = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    y = np.array([mapper[t] for t in y])

    # Split data to train and test.
    np.random.seed(1)
    data_index = np.arange(150)
    np.random.shuffle(data_index)
    train, test = data_index[:120], data_index[120:]
    print(train, test)

    mlp = MLP(4, 1, 3, 3)
    mlp.fit(X[train], y[train])
    print('Predict:', np.argmax(mlp.predict(X[test]), axis=1))
    print('Target: ', np.argmax(y[test], axis=1))

    # c = [2, 3]
    # plt.scatter(iris['data'][:50, c[0]], iris['data'][:50, c[1]], c='r', marker='.', label='setoia')
    # plt.scatter(iris['data'][50:100, c[0]], iris['data'][50:100, c[1]], c='b', marker='.', label='setoia')
    # plt.scatter(iris['data'][100:150, c[0]], iris['data'][100:150, c[1]], c='g', marker='.', label='setoia')
    # plt.show()


    # X = np.array([
    #                  [1, 0],
    #                  [0, 1],
    #              ] * 1000)
    # y = np.array([
    #                  [0, 1],
    #                  [1, 0],
    #              ] * 1000)
    # mlp = MLP(2, 5, 5, 2)
    # mlp._verbosity = 0
    # mlp.fit(X, y)
    # mlp._verbosity = 2
    # mlp.predict(X)


if __name__ == '__main__':
    main()
