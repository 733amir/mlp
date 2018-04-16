import numpy as np
from math import ceil
from utils import Utils


class Trainer:

    def __init__(self, model, X, y, test_percentage=0.2, random_state=1, y_is_one_hot=False):
        self._model = model
        self._test_percentage = test_percentage
        self._random_state = random_state

        # Convert target class to one-hot list of classes.
        if not y_is_one_hot:
            targets = np.unique(y)
            new_y = np.zeros((len(y), len(targets)))
            for i, t in np.ndenumerate(targets):
                new_y[y == t, i] = 1
            y = new_y

        # Split data to train and test.
        index = np.arange(X.shape[0])
        np.random.seed(self._random_state)
        np.random.shuffle(index)
        boundry = int(np.ceil(index.shape[0] * self._test_percentage))
        test_index, train_index = index[:boundry], index[boundry:]

        self._train_X, self._test_X = X[train_index], X[test_index]
        self._train_y, self._test_y = y[train_index], y[test_index]

        Utils.log('Splitted data (train_x, train_y, test_x, test_y):',
                  [self._train_X, self._train_y, self._test_X, self._test_y], verbosity=1)

    def train(self):
        self._model.fit(self._train_X, self._train_y)

    def kfold(self, number_of_folds=10):
        dataset_size = len(self._train_X)
        fold_size = ceil(dataset_size / 10)

        accuracy = []
        weights = []
        biases = []
        for boundry in range(0, dataset_size, fold_size):
            start, end = boundry, min(boundry + fold_size, dataset_size)

            eval_X = self._train_X[start:end]
            eval_y = self._train_y[start:end]
            remain_X = np.concatenate((self._train_X[:start], self._train_X[end:]), axis=0)
            remain_y = np.concatenate((self._train_y[:start], self._train_y[end:]), axis=0)

            Utils.log(f'{number_of_folds}-fold data (eval_x, eval_y, remain_x, remain_y):',
                      [eval_X, eval_y, remain_X, remain_y], verbosity=1)

            self._model._weights = self._model._initialize_weights(*self._model._nn_dimensions)
            self._model._biases = self._model._initialize_biases(*self._model._nn_dimensions)

            self._model.fit(remain_X, remain_y)
            predicted = self._model.predict(eval_X)

            accuracy.append((predicted == eval_y).sum() / len(predicted))
            weights.append(self._model._weights)
            biases.append(self._model._biases)

        Utils.log(f'{number_of_folds}-fold weights, bias and accuracy:', [weights, biases, accuracy], verbosity=1)

        avg_weights = self._get_avg(weights)
        avg_biases = self._get_avg(biases)
        avg_accuracy = sum(accuracy) / len(accuracy)

        self._model._weights = avg_weights
        self._model._biases = avg_biases

        return avg_accuracy

    def run_test(self):
        predicted = self._model.predict(self._test_X)
        correct = np.argmax(predicted, axis=1) == np.argmax(self._test_y, axis=1)
        return (correct.sum(), len(correct))

    @staticmethod
    def _get_avg(data):
        avg_data = []
        for i, ds in enumerate(data):
            for j, d in enumerate(ds):
                if len(avg_data) <= j:
                    avg_data.append(d)
                else:
                    avg_data[j] += d
        return [d / len(data) for d in avg_data]
