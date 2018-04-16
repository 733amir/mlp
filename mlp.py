from activation_functions import ActivationFunctions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from trainer import Trainer
from utils import Utils


class Constants:
    DIMENSIONS = 'DIMENSIONS'
    ACTIVATION_FUNCTIONS = 'ACTIVATION_FUNCTIONS'
    RANDOM_STATE = 'RANDOM_STATE'
    WEIGHTS = 'WEIGHTS'
    BIASES = 'BIASES'

    DELTA_ERR = 'DELTA_ERR'
    EPOCH = 'EPOCH'


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
    _normal_parameters = [0, 0.1]

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

        self._nn_activation_funcs = ActivationFunctions.init_linear_sigmoids_softmax(hidden_layers)

        # Initializing NumPy random number generator.
        self._random_state = random_state
        self._random_generator = np.random.RandomState(self._random_state)

        # Initializing weights and biases of our model using Normal distribution with mean and variance configurable.
        self._weights = self._initialize_weights(*self._nn_dimensions)
        self._biases = self._initialize_biases(*self._nn_dimensions)

        Utils.log('Initialized weights: [layer][neuron, weight]', self._weights, verbosity=2)
        Utils.log('Initialized biases: [layer][bias]', self._biases, verbosity=2)

        # Used variables during training.
        self._inputs, self._outputs = [], []
        self._errors = []

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

    def predict(self, X, one_hot=True):
        """
        Calculate output using weights, biases and activation functions of the model.

        :param X: Is a NumPy array representing each observation in a row. It can contains multiple observations.
        :return: A NumPy array containing the result of the observation. Number of columns is equal to output (last)
            layer and number of rows is equal to number of observations (rows) in input data.
        """
        Utils.log(f'Input for prediction to Network:', [X], verbosity=2)

        self._inputs, self._outputs = [], []
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            self._inputs.append(X.dot(w) + b)
            X = self._nn_activation_funcs[i](self._inputs[-1])
            self._outputs.append(X)

            # Log each batch of calculations.
            Utils.log(f'Layer #{i + 1} weights (column => neuron):', [w], verbosity=3)
            Utils.log(f'Layer #{i + 1} bias (column => neuron):', [b], verbosity=3)
            Utils.log(f'Layer #{i + 1} net input:', [self._inputs[-1]], verbosity=3)
            Utils.log(f'Layer #{i + 1} output ({self._nn_activation_funcs.get_name(i)}):', [X], verbosity=3)

        Utils.log(f'Output of Network prediction:', [X], verbosity=2)

        if one_hot:
            result = np.zeros(X.shape)
            for i, max_index in np.ndenumerate(np.argmax(X, axis=1)):
                result[i, max_index] = 1
            return result
        else:
            return X

    def fit(self, X, y, epoch=1000, eta=0.1, batch_size=50, delta_err=0.001, stop_condition=Constants.EPOCH):
        """
        Update the model with provided data.

        :param X: Input for the network.
        :param y: Desired output.
        :param epoch: Number of looping on provided data and updating model.
        :param eta: Learning rate.
        :param batch_size: Break provided data to batches of this size.
        :param delta_err: Delta error between each batch update of model.
        :param stop_condition: Chossing between Epoch and Delta Error as stop condition.
        :return: Error that each batch update had.
        """
        self._errors = []

        if stop_condition == Constants.EPOCH:
            for i in range(epoch):
                for j in range(0, X.shape[0], batch_size):
                    start, end = j, min(j + batch_size, X.shape[0])
                    batch_X, batch_y = X[start:end], y[start:end]
                    self._errors.append(self._batch_update(batch_X, batch_y, eta))

        elif stop_condition == Constants.DELTA_ERR:
            while True:
                for j in range(0, X.shape[0], batch_size):
                    start, end = j, min(j + batch_size, X.shape[0])
                    batch_X, batch_y = X[start:end], y[start:end]
                    self._errors.append(self._batch_update(batch_X, batch_y, eta))

                    if len(self._errors) >= 2 and self._errors[-2] - self._errors[-1] <= delta_err:
                        break
                if len(self._errors) >= 2 and self._errors[-2] - self._errors[-1] <= delta_err:
                    break

        return self._errors

    def _batch_update(self, X, y, eta):
        """
        Update the model with batch update.

        :param X: Input for the network.
        :param y: Desired output.
        :param eta: Learning rate.
        :return: Error of the network.
        """
        self.predict(X, one_hot=False)
        self._delta = self._initialize_delta(*self._nn_dimensions)

        dmse = y - self._outputs[-1]
        mse = (dmse ** 2 / 2).sum()
        Utils.log('Mean Squared Error:', [mse], verbosity=1)

        Utils.log('Weights', [w.shape for w in self._weights], verbosity=4)
        Utils.log('Delta', [d.shape for d in self._delta], verbosity=4)
        Utils.log('Output', [o.shape for o in self._outputs], verbosity=4)
        Utils.log('d MSE', [dmse.shape], verbosity=4)

        self._delta[-1] = dmse * (self._outputs[-1] * (1 - self._outputs[-1]))  # TODO Generalize

        for i in range(-2, -len(self._delta) - 1, -1):
            self._delta[i] = self._delta[i + 1].dot(self._weights[i + 1].T) * \
                             (self._outputs[i] * (1 - self._outputs[i]))  # TODO Generalize

        Utils.log('Delta matrix:', self._delta, verbosity=2)

        # Updating Parameters
        for i in range(-1, -len(self._delta), -1):
            self._weights[i] += (eta / self._delta[i].shape[0]) * self._outputs[i - 1].T.dot(self._delta[i])
            self._biases[i] += (eta / self._delta[i].shape[0]) * self._delta[i].sum(axis=0)

        Utils.log('Weights after update:', self._weights, verbosity=3)
        Utils.log('Biases after update:', self._biases, verbosity=3)

        return mse

    def save_to_file(self, path):
        """
        Write a file containing parameters of the model.

        :param path: A path to a file.
        """
        with open(path, 'w') as storage:
            storage.write(str(self._get_parameters()))

    @classmethod
    def load_from_file(cls, path):
        """
        Read a file containing parameters of a model and load them.

        :param path: A path to a file containing parameters of a model.
        :return: Generated model.
        """
        with open(path, 'r') as storage:
            params = eval(storage.read())
            return cls._set_parameters(params)

    def _get_parameters(self):
        """
        Save all parameters of the model to a dictionary.

        :return: Dictionary containing parameters of the model.
        """
        return {
            Constants.DIMENSIONS: self._nn_dimensions,
            Constants.ACTIVATION_FUNCTIONS: self._nn_activation_funcs.get_names(),
            Constants.RANDOM_STATE: self._random_state,
            Constants.WEIGHTS: [w.tolist() for w in self._weights],
            Constants.BIASES: [b.tolist() for b in self._biases],
        }

    @classmethod
    def _set_parameters(cls, params):
        """
        Get a dictionary containing parameters of a model and generate the model.

        :param params: Dictionary containing parameters of the model.
        :return: Generated model using dictionary containing parameters of the model.
        """
        new_mlp = cls(*params[Constants.DIMENSIONS], random_state=params[Constants.RANDOM_STATE])

        new_mlp._nn_activation_funcs = ActivationFunctions(params[Constants.ACTIVATION_FUNCTIONS])
        new_mlp._weights = [np.array(w) for w in params[Constants.WEIGHTS]]
        new_mlp._biases = [np.array(b) for b in params[Constants.BIASES]]

        return new_mlp

    def _normal_random(self, size):
        """
        Generating a NumPy ndarray in shape of `size` and filling it with random numbers from a normal distribution.

        :param size: Shape of the NumPy ndarray.
        :return: NumPy ndarray containing random number from normal distribution.
        """
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

    t = Trainer(MLP(4, 1, 3, 3), X, y)
    t.train()
    print(t.run_test())


    # # Converting target to one-hot target.
    # mapper = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # y = np.array([mapper[t] for t in y])
    #
    # # Split data to train and test.
    # np.random.seed(1)
    # data_index = np.arange(150)
    # np.random.shuffle(data_index)
    # train, test = data_index[:120], data_index[120:]
    # print(train, test)
    #
    # mlp = MLP(4, 1, 3, 3)
    # mlp.fit(X[train], y[train])
    # mlp.save_to_file('test')
    # mlp = MLP.load_from_file('test')
    # print('Predict:', np.argmax(mlp.predict(X[test]), axis=1))
    # print('Target: ', np.argmax(y[test], axis=1))

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
