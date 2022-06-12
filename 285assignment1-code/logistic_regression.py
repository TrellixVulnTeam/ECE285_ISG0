"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        s = 1 / (1 + np.exp(-z))
        return s
        # TODO: implement me
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights
        
        for m in range(10):
            for j in range(self.n_class):
                g = np.zeros(D)
                y = np.where(y_train == j, 1, -1)
                for i in range(N):
                    m = (-1/N)*self.sigmoid((-1)*y[i]*self.w[j,:]@X_train[i,:].T)*y[i]*X_train[i,:]
                    g = g + m

                self.w[j, :] = self.w[j, :] - self.lr * g - self.lr * self.weight_decay * self.w[j, :]
        # TODO: implement me

        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        y_hat = X_test @ self.w.T
        y_max = np.zeros(X_test.shape[0])
        y_max = y_hat.argmax(axis=1)
        return y_max
        pass
