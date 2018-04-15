import numpy as np


class ActivationFunctions:
    """
    A class to manage activation function for each layer and they're derivatives.
    """
    Linear = 'Linear'
    Sigmoid = 'Sigmoid'
    Softmax = 'Softmax'

    def __init__(self, funcs):
        self._activation_funcs = {
            'Linear': self.linear,
            'Sigmoid': self.sigmoid,
            'Softmax': self.softmax,
        }

        self._activation_funcs_derivative = {
            'Linear': self.linear_derivative,
            'Sigmoid': self.sigmoid_derivative,
            'Softmax': self.softmax_derivative,
        }

        self._funcs = funcs

    @classmethod
    def init_linear_sigmoids_softmax(cls, number_of_hidden_layers):
        return cls([cls.Linear] + [cls.Sigmoid] * number_of_hidden_layers + [cls.Softmax])

    def __getitem__(self, key):
        return self._activation_funcs[self._funcs[key]]

    def __setitem__(self, key, value):
        self._funcs[key] = value

    def get_name(self, key):
        return self._funcs[key]

    def get_derivative(self, key):
        return self._activation_funcs_derivative[self._funcs[key]]

    def get_names(self):
        return self._funcs

    @staticmethod
    def linear(X):
        """
        Simple Linear function returning what gets in the input.

        :param X: NumPy array consisting the input.
        :return: Input without any change.
        """
        return X

    @staticmethod
    def linear_derivative(X):
        """
        Simple derivative of Linear function returning what gets in the input.

        :param X: NumPy array consisting the input.
        :return: Input without any change.
        """
        return X

    @staticmethod
    def sigmoid(X):
        """
        Calculate Sigmoid function on NumPy array data.

        :param X: NumPy array consisting the input.
        :return: NumPy array consisting output of Sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-X))

    @staticmethod
    def sigmoid_derivative(X):
        """
        Calculate derivative of Sigmoid function on NumPy array data.

        :param X: NumPy array consisting the input.
        :return: NumPy array consisting output of derivative of Sigmoid function.
        """
        X = ActivationFunctions.sigmoid(X)
        return X * (1 - X)

    @staticmethod
    def softmax(X):
        """
        Calculate Softmax function on NumPy array data.

        :param X: NumPy array consisting the input.
        :return: NumPy array consisting output of Softmax function.
        """
        e = np.exp(X)
        return e / np.sum(e, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(X1, X2=None):
        """
        Calculate derivative of Softmax function on NumPy array data.

        :param X: NumPy array consisting the input.
        :param same: Derivative function against the input of it.
        :return: NumPy array consisting output of derivative of Softmax function.
        """
        X1 = ActivationFunctions.softmax(X1)

        if X2 is not None:
            X2 = ActivationFunctions.softmax(X2)
            return -X1 * X2

        return X1 * (1 - X1)
