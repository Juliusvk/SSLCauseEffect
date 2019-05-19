"""
This script contains code to run experiments for semi-supervised learning (SSL) with cause and effect features.
"""

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge


def sigmoid(x):
    """
    computes the logistic sigmoid function evaluated at input x
    :param x: (n x 1) np.array of reals
    :return: (n x 1) np.array of probabilities
    """

    return 1/(1 + np.exp(-x))


def fy_linear(x, a, b):
    """
    computes the linear function x*W + b , e.g. the logit class probs (the input to the sigmoid) from causal features
    :param x: (n x d) np.array of reals - causal features X_C
    :param a: (d x p) weight matrix
    :param b: (1 x p) bias term
    :return: (n x p) np.array of reals
    """

    return np.matmul(x, a) + b


def sample_from_mog(weights, means, covs, n_samples):
    """
    Generates samples from a d-dimensional mixture of Gaussians
    :param weights: (m x 1) np.array - weights of mixture components which have to sum to 1
    :param means: (m x d) np.array of means
    :param covs: (m x d x d) np.array of covariances
    :param n_samples: int number of samples to be drawn
    :return: (n_samples x d) np.array of samples from d-dimensional mixture of Gaussians
    """

    d = means.shape[1]
    comps = np.random.multinomial(1, weights, n_samples)  # (n_samplesxm) mask of components
    sample_means = np.einsum('ij,jkl->ikl', comps, means)  # (n_samplesxd) matrix of sample means
    sample_covs = np.einsum('ij,jkl->ikl', comps, covs)  # (n_samplesxdxd) tensor of sample variances
    samples = np.zeros((n_samples, d))
    for i in range(n_samples):
        samples[i] = np.random.multivariate_normal(sample_means[i].ravel(), sample_covs[i])
    return samples


def get_data_linear(weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_samples):
    """
    Generates a synthetic data set of size n_samples according to the generative model:

    X_C ~ MoG(weights_c, means_c, covs_c),      X_C is in R^d_c
    Y := I[sigmoid(x_C * a_y + b_y > N_Y],      N_Y ~ U[0,1],       Y is in {0,1}
    X_E := a_e * X_C + b_1*Y + b_0*(1-Y) + N_E,     N_E ~ N(0, cov_e),      X_E is in R^d_e

    :param weights_c: (m x 1) np.array - weights of mixture components which have to sum to 1
    :param means_c: (m x d_c) np.array of means
    :param covs_c: (m x d_c x d_c) np.array of covariances
    :param a_y: (d_c x 1) np.array of weights for logistic regression of Y on X_C
    :param b_y: (1 x 1) bias term for logistic regression of Y on X_C
    :param a_e0: (d_c x d_e) np.array of weights for map X_C, Y=0 -> X_E
    :param a_e1: (d_c x d_e) np.array of weights for map X_C, Y=1 -> X_E
    :param b_0: (1 x d_e) np.array bias for class Y=0
    :param b_1: (1 x d_e) np.array bias for class Y=1
    :param cov_e0: (d_e x d_e) np.array covariance for noise  N_E | Y=0
    :param cov_e1: (d_e x d_e) np.array covariance for noise  N_E | Y=1
    :param n_samples:
    :return: x_c: (n_samples x d_c) np.array of causal features
             y: (n_samples x 1) np.array of class labels
             x_e: (n_samples x d_e) np.array of effect features
    """

    x_c = sample_from_mog(weights_c, means_c, covs_c, n_samples)
    class_probs = sigmoid(fy_linear(x_c, a_y, b_y))  # P(Y=1 | X_C)
    n_y = np.random.uniform(0, 1, (n_samples, 1))
    y = np.ones((n_samples, 1)) * (class_probs > n_y)
    d_e = cov_e0[0].shape
    n_e0 = np.random.multivariate_normal(np.zeros(d_e), cov_e0, n_samples)
    n_e1 = np.random.multivariate_normal(np.zeros(d_e), cov_e1, n_samples)
    x_e0 = np.matmul(x_c, a_e0) + b_0 + n_e0
    x_e1 = np.matmul(x_c, a_e1) + b_1 + n_e1
    x_e = np.multiply(y == 0, x_e0) + np.multiply(y == 1, x_e1)
    return x_c, y, x_e


def predict_class_probs(x_c, x_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1):
    """
    Assigns examples of the form (x_c, x_e) to the more likely class given the (estimated) parameters.
    :param x_c: (n x d_c) array of causal features
    :param x_e: (n x d_e) array of effect features
    :param a_y: (d_c x 1) np.array of weights for logistic regression of Y on X_C
    :param b_y: (1 x 1) bias term for logistic regression of Y on X_C
    :param a_e0: (d_c x d_e) np.array of weights for map X_C, Y=0 -> X_E
    :param a_e1: (d_c x d_e) np.array of weights for map X_C, Y=1 -> X_E
    :param b_0: (1 x d_e) np.array bias for class Y=0
    :param b_1: (1 x d_e) np.array bias for class Y=1
    :param cov_e0: (d_e x d_e) np.array covariance for noise  N_E |Y=0
    :param cov_e1: (d_e x d_e) np.array covariance for noise  N_E |Y=1
    :return: (n x 1) array of class probabilities representing P(Y=1 | X_C, X_E)
    """

    py1 = sigmoid(fy_linear(x_c, a_y, b_y))  # P(Y=1 |X_C)

    mean0 = np.matmul(x_c, a_e0) + b_0
    pe0 = np.zeros(py1.shape)
    for i in range(py1.shape[0]):
        pe0[i] = multivariate_normal.pdf(x_e[i], mean0[i], cov_e0)  # P(X_E| X_C, Y=0)

    mean1 = np.matmul(x_c, a_e1) + b_1
    pe1 = np.zeros(py1.shape)
    for i in range(py1.shape[0]):
        pe1[i] = multivariate_normal.pdf(x_e[i], mean1[i], cov_e1)  # P(X_E| X_C, Y=1)

    prelim = np.multiply(py1, pe1)
    p1 = np.divide(prelim, prelim + np.multiply(1-py1, pe0))
    return p1


def plot_data(x_c, y, x_e, z_c, z_e):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(z_c, z_e, color='grey', marker='.')
    ax.scatter(x_c[y == 0], x_e[y == 0], color='blue', marker='.')
    ax.scatter(x_c[y == 1], x_e[y == 1], color='red', marker='.')
    ax.set(xlabel='Causal features $X_C$', ylabel='Effect features $X_E$')
    # ax.legend(loc='best')
    # plt.show()
    return fig


def get_log_reg_params(x, y, weight=None):
    lr_c_only = LogisticRegression(random_state=0, solver='liblinear')
    lr_c_only.fit(x, y.ravel(), sample_weight=weight)
    return np.transpose(lr_c_only.coef_), lr_c_only.intercept_


def get_weighted_lin_reg_params(x_c, x_e, w):
    ridge = Ridge().fit(x_c, x_e, sample_weight=w)
    a = np.transpose(ridge.coef_)
    b = ridge.intercept_.reshape((1, x_e.shape[1]))
    sq_res = np.square(x_e - ridge.predict(x_c))
    sum_weighted_sq_res = np.sum(np.multiply(w.reshape((w.shape[0], 1)), sq_res), axis=0)
    cov = np.diag(np.divide(sum_weighted_sq_res, np.sum(w)))
    return a, b, cov


def get_lin_reg_params(x_c, x_e):
    ridge = Ridge().fit(x_c, x_e)
    a = np.transpose(ridge.coef_)
    b = ridge.intercept_.reshape((1, x_e.shape[1]))
    sum_sq_res = np.sum(np.square(x_e - ridge.predict(x_c)), axis=0)
    cov = np.diag(np.divide(sum_sq_res, x_c.shape[0]))
    return a, b, cov


def hard_label_EM(x_c, y, x_e, z_c, z_e):
    c = np.concatenate((x_c, z_c))
    e = np.concatenate((x_e, z_e))

    # initialise from labelled data
    a_y, b_y = get_log_reg_params(x_c, y)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    a_e0, b_0, cov_e0 = get_lin_reg_params(x_c[idx_0], x_e[idx_0])
    a_e1, b_1, cov_e1 = get_lin_reg_params(x_c[idx_1], x_e[idx_1])

    converged = False
    u_old = np.zeros((z_c.shape[0], 1))
    while not converged:
        # E-step: compute labels
        p1 = predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)
        u = p1 > 0.5

        # Check for convergence
        if (u_old == u).all():
            break

        # M-step:
        l = np.concatenate((y, u))
        a_y, b_y = get_log_reg_params(c, l)
        a_e0, b_0, cov_e0 = get_weighted_lin_reg_params(c, e, 1-l.ravel())
        a_e1, b_1, cov_e1 = get_weighted_lin_reg_params(c, e, l.ravel())
        u_old = u

    return a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1


def soft_label_EM(x_c, y, x_e, z_c, z_e, tol=1e-3):
    c = np.concatenate((x_c, z_c))
    e = np.concatenate((x_e, z_e))

    # initialise from labelled data
    a_y, b_y = get_log_reg_params(x_c, y)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    a_e0, b_0, cov_e0 = get_lin_reg_params(x_c[idx_0], x_e[idx_0])
    a_e1, b_1, cov_e1 = get_lin_reg_params(x_c[idx_1], x_e[idx_1])

    converged = False
    u_old = np.zeros((z_c.shape[0], 1))
    while not converged:
        # E-step: compute labels
        p1 = predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)
        p = np.concatenate((y, p1)) # probabilities for class 1
        u = p1 > 0.5
        l = np.concatenate((y, u))
        w = (l * p + (1-l) * (1-p)).ravel()

        # M-step:
        a_y, b_y = get_log_reg_params(c, l, w)
        a_e0, b_0, cov_e0 = get_weighted_lin_reg_params(c, e, 1-p.ravel())
        a_e1, b_1, cov_e1 = get_weighted_lin_reg_params(c, e, p.ravel())

        # Check for convergence
        if max(np.abs(u_old-p1)) < tol:
            converged = True
        else:
            u_old = p1

    return a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1
