"""
This script contains code to run experiments for semi-supervised learning (SSL) with cause and effect features.
"""
import numpy as np


def sigmoid(x):
    """
    computes the logistic sigmoid function evaluated at input x
    :param x: (nx1) np.array of reals
    :return: (nx1) np.array of probabilities
    """
    return 1/(1 + np.exp(-x))


def fy_linear(x, a, b):
    """
    computes the linear function x*W + b , e.g. the logit class probabilities (the input to the sigmoid) from causal features
    :param x: (nxd) np.array of reals - causal features X_C
    :param a: (dxp) weight matrix
    :param b: (1xp) bias term
    :return: (nxp) np.array of reals
    """
    return np.matmul(x, a) + b


# x_c = 0.5 * np.ones((10, 3))
# a_c = 0.3 * np.ones((3, 1))
# b_c = 0.7 * np.ones((1, 1))
# print(sigmoid(fy_linear(x_c, a_c, b_c)))


def sample_from_mog(weights, means, covs, n_samples):
    """
    Generates samples from a d-dimensional mixture of Gaussians
    :param weights: (mx1) np.array - weights of mixture components which have to sum to 1
    :param means: (mxd) np.array of means
    :param covs: (mxdxd) np.array of covariances
    :param n_samples: int number of samples to be drawn
    :return: (n_samplesxd) np.array of samples from d-dimensional mixture of Gaussians
    """
    d = means.shape[1]
    comps = np.random.multinomial(1, weights, n_samples)  # (n_samplesxm) mask of components
    sample_means = np.matmul(comps, means)  # (n_samplesxd) matrix of sample means
    sample_covs = np.einsum('ij,jkl->ikl', comps, covs)  # (n_samplesxdxd) tensor of sample variances
    samples = np.zeros((n_samples, d))
    for i in range(n_samples):
        samples[i] = np.random.multivariate_normal(sample_means[i], sample_covs[i])
    return samples


# m = 3
# d = 2
# n_samples = 10
# weights = np.array([.3, .2, .5])
# means = np.zeros((m, d))
# covs = np.zeros((m, d, d))
# for i in range(m):
#     covs[i] = np.eye(d)
# print(sample_from_mog(weights,means, covs, n_samples))

