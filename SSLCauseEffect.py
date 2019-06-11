"""
This script contains code to run experiments for semi-supervised learning (SSL) with cause and effect features.
"""

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from scikitTSVM import SKTSVM


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


def get_log_reg_params(x, y, weight=None):
    lr_c_only = LogisticRegression(random_state=0, solver='liblinear')
    lr_c_only.fit(x, y.ravel(), sample_weight=weight)
    return np.transpose(lr_c_only.coef_), lr_c_only.intercept_


def get_weighted_lin_reg_params(x_c, x_e, w, lam=1e-3):
    ridge = Ridge().fit(x_c, x_e, sample_weight=w)
    a = np.transpose(ridge.coef_)
    b = ridge.intercept_.reshape((1, x_e.shape[1]))
    sq_res = np.square(x_e - ridge.predict(x_c))
    sum_weighted_sq_res = np.sum(np.multiply(w.reshape((w.shape[0], 1)), sq_res), axis=0)
    cov = np.diag(np.divide(sum_weighted_sq_res, np.sum(w)))
    cov += np.diag(lam * np.ones((x_e.shape[1], x_e.shape[1])))
    if np.linalg.matrix_rank(cov) < x_e.shape[1]:
        print 'MATRIX SINGULAR IN WEIGHTED REG'
    return a, b, cov


def get_lin_reg_params(x_c, x_e, lam=1e-3):
    ridge = Ridge().fit(x_c, x_e)
    a = np.transpose(ridge.coef_)
    b = ridge.intercept_.reshape((1, x_e.shape[1]))
    sum_sq_res = np.sum(np.square(x_e - ridge.predict(x_c)), axis=0)
    cov = np.diag(np.divide(sum_sq_res, x_c.shape[0]))
    cov += np.diag(lam * np.ones((x_e.shape[1], x_e.shape[1])))
    if np.linalg.matrix_rank(cov) < x_e.shape[1]:
        print 'MATRIX SINGULAR'
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


def soft_label_EM(x_c, y, x_e, z_c, z_e, converged=False, tol=1e-2):
    c = np.concatenate((x_c, z_c))
    e = np.concatenate((x_e, z_e))

    # initialise from labelled data
    a_y, b_y = get_log_reg_params(x_c, y)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    a_e0, b_0, cov_e0 = get_lin_reg_params(x_c[idx_0], x_e[idx_0])
    a_e1, b_1, cov_e1 = get_lin_reg_params(x_c[idx_1], x_e[idx_1])

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


def conditional_prop(x_c, y, x_e, z_c, z_y, z_e):
    y_unl = -1 * np.ones((z_c.shape[0],1))
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    c_0 = x_c[idx_0]
    c_1 = x_c[idx_1]
    e_0 = x_e[idx_0]
    e_1 = x_e[idx_1]

    # initialise from labelled data
    R0 = Ridge().fit(c_0, e_0)
    R1 = Ridge().fit(c_1, e_1)

    while sum(y_unl == -1) > 0:
        # check what is still unlabelled
        idx_remaining = np.where(y_unl == -1)[0]

        # predict for unlabelled points
        err_0 = np.sum(np.square(z_e[idx_remaining] - R0.predict(z_c[idx_remaining])), axis=1)
        err_1 = np.sum(np.square(z_e[idx_remaining] - R1.predict(z_c[idx_remaining])), axis=1)

        # assign best fitting point to correct label
        if min(err_0) < min(err_1):
            # update R0
            idx = idx_remaining[np.argmin(err_0)]
            y_unl[idx] = 0
            np.append(c_0, z_c[idx])
            np.append(e_0, z_e[idx])
            R0 = Ridge().fit(c_0, e_0)
        else:
            # update R1
            idx = idx_remaining[np.argmin(err_1)]
            y_unl[idx] = 1
            np.append(c_1, z_c[idx])
            np.append(e_1, z_e[idx])
            R1 = Ridge().fit(c_1, e_1)

    return np.mean(y_unl == z_y)


def discrete_data_EM(x_c, y, x_e, z_c, z_y, z_e):
    c = np.concatenate((x_c, z_c))
    e = np.concatenate((x_e, z_e))
    LRC = LogisticRegression(random_state=0, solver='liblinear')
    LRC.fit(x_c, y.ravel())
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    LR0 = LogisticRegression(random_state=0, solver='liblinear')
    LR0.fit(x_c[idx_0], x_e[idx_0])
    LR1 = LogisticRegression(random_state=0, solver='liblinear')
    LR1.fit(x_c[idx_1], x_e[idx_1])
    pass


def run_methods(x_c, y, x_e, z_c, z_y, z_e):
    x = np.concatenate((x_c, x_e), axis=1)
    z = np.concatenate((z_c, z_e), axis=1)

    # Baseline: Linear Logistic Regression
    lin_lr = LogisticRegression(random_state=0, solver='liblinear').fit(x, y.ravel())
    acc_lin_lr = lin_lr.score(z, z_y)
    # hard_label_lin_lr = lin_lr.predict(z)
    # soft_label_lin_lr = lin_lr.predict_proba(z)[:, 1]

    # TRANSDUCTIVE APPROACHES
    # merge labelled and unlabelled data (with label -1) for transductive methods
    x_merged = np.concatenate((x, z))
    y_merged = np.concatenate((y, -1 * np.ones((z.shape[0], 1)))).ravel().astype(int)

    # Baseline: Linear TSVM: https://github.com/tmadl/semisup-learn/tree/master/methods
    lin_tsvm = SKTSVM(kernel='linear')
    lin_tsvm.fit(x_merged, y_merged)
    acc_lin_tsvm = lin_tsvm.score(z, z_y)
    # hard_label_lin_tsvm = lin_tsvm.predict(z)
    # soft_label_lin_tsvm = lin_tsvm.predict_proba(z)[:, 1]

    # Baseline: Non-Linear TSVM:  https://github.com/tmadl/semisup-learn/tree/master/methods
    rbf_tsvm = SKTSVM(kernel='RBF')
    rbf_tsvm.fit(x_merged, y_merged)
    acc_rbf_tsvm = rbf_tsvm.score(z, z_y)
    # hard_label_rbf_tsvm = rbf_tsvm.predict(z)
    # soft_label_rbf_tsvm = rbf_tsvm.predict_proba(z)[:, 1]

    # Baseline: Label Propagation RBF weights
    try:
        rbf_label_prop = LabelPropagation(kernel='rbf')
        rbf_label_prop.fit(x_merged, y_merged)
        acc_rbf_label_prop = rbf_label_prop.score(z, z_y)
        # hard_label_rbf_label_prop= rbf_label_prop.predict(z)
        # soft_label_rbf_label_prop = rbf_label_prop.predict_proba(z)[:, 1]
    except:
        acc_rbf_label_prop = []
        print 'rbf label prop did not work'

    # Baseline: Label Spreading with RBF weights
    try:
        rbf_label_spread = LabelSpreading(kernel='rbf')
        rbf_label_spread.fit(x_merged, y_merged)
        acc_rbf_label_spread = rbf_label_spread.score(z, z_y)
        # hard_label_rbf_label_spread = rbf_label_spread.predict(z)
        # soft_label_rbf_label_spread = rbf_label_spread.predict_proba(z)[:, 1]
    except:
        acc_rbf_label_spread = []
        print 'rbf label spread did not work '

    # THE K-NN VERSIONS ARE UNSTABLE UNLESS USING LARGE K
    # Baseline: Label Propagation with k-NN weights
    try:
        knn_label_prop = LabelPropagation(kernel='knn', n_neighbors=11)
        knn_label_prop.fit(x_merged, y_merged)
        acc_knn_label_prop = knn_label_prop.score(z, z_y)
        # hard_label_knn_label_prop = knn_label_prop.predict(z)
        # soft_label_knn_label_prop = knn_label_prop.predict_proba(z)[:, 1]
    except:
        acc_knn_label_prop = []
        print 'knn label prop did not work'

    # Baseline: Label Spreading with k-NN weights
    try:
        knn_label_spread = LabelSpreading(kernel='knn', n_neighbors=11)
        knn_label_spread.fit(x_merged, y_merged)
        acc_knn_label_spread = knn_label_spread.score(z, z_y)
        # hard_label_knn_label_spread = knn_label_spread.predict(z)
        # soft_label_knn_label_spread = knn_label_spread.predict_proba(z)[:, 1]
    except:
        acc_knn_label_spread = []
        print 'knn label spread did not work'

    # Generative Models
    # Semi-generative model on labelled data only
    a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = soft_label_EM(x_c, y, x_e, z_c, z_e, converged=True)
    soft_label_semigen = predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)
    hard_label_semigen = soft_label_semigen > 0.5
    acc_semigen_labelled = np.mean(hard_label_semigen == z_y)

    # EM with soft labels
    a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = soft_label_EM(x_c, y, x_e, z_c, z_e)
    soft_label_soft_EM = predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)
    hard_label_soft_EM = soft_label_soft_EM > 0.5
    acc_soft_EM = np.mean(hard_label_soft_EM == z_y)

    # EM with hard labels
    a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = hard_label_EM(x_c, y, x_e, z_c, z_e)
    soft_label_hard_EM = predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)
    hard_label_hard_EM = soft_label_hard_EM > 0.5
    acc_hard_EM = np.mean(hard_label_hard_EM == z_y)

    # Conditional label prop
    acc_cond_prop = conditional_prop(x_c, y, x_e, z_c, z_y, z_e)

    return acc_lin_lr, acc_lin_tsvm, acc_rbf_tsvm, acc_rbf_label_prop, acc_rbf_label_spread, acc_knn_label_prop,\
           acc_knn_label_spread, acc_semigen_labelled, acc_soft_EM, acc_hard_EM, acc_cond_prop