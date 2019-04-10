"""
This script contains code to run experiments for semi-supervised learning (SSL) with cause and effect features.
"""
import numpy as np


def sigmoid(x):
    """
    computes the logistic sigmoid function of evaluated at input
    :param x: (dx1) np.array of reals
    :return: (dx1) np.array of probabilities
    """
    return 1/(1 + np.exp(-x))


a = np.array([2.3, 0, -4])
print(sigmoid(a))