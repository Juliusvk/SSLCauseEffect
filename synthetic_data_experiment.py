import SSLCauseEffect as ssl
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# Generate Synthetic Datasets
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
    X_E := (a_e1 * X_C + b_1) * Y + (a_e0 * X_C + b_0) * (1-Y) + N_E,     N_E ~ N(0, cov_e),      X_E is in R^d_e

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
    # ensure at least 2 samples per class
    n_0 = 0
    n_1 = 0
    while n_0 < 2 or n_1 < 2:
        x_c = sample_from_mog(weights_c, means_c, covs_c, n_samples)
        class_probs = ssl.sigmoid(ssl.fy_linear(x_c, a_y, b_y))  # P(Y=1 | X_C)
        n_y = np.random.uniform(0, 1, (n_samples, 1))
        y = np.ones((n_samples, 1)) * (class_probs > n_y)
        n_0 = sum(y == 0)
        n_1 = sum(y == 1)

    d_e = cov_e0[0].shape
    n_e0 = np.random.multivariate_normal(np.zeros(d_e), cov_e0, n_samples)
    n_e1 = np.random.multivariate_normal(np.zeros(d_e), cov_e1, n_samples)
    x_e0 = np.matmul(x_c, a_e0) + b_0 + n_e0
    x_e1 = np.matmul(x_c, a_e1) + b_1 + n_e1
    x_e = np.multiply(y == 0, x_e0) + np.multiply(y == 1, x_e1)
    return x_c, y, x_e


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


# Change parameters in here to generate different synthetic datasets.
def get_params(d_c, d_e):
    """
    Sets the parameters for different synthetic datasets
    :param d_c: dimension of causal features X_C
    :param d_e: dimension of effect features X_E
    :return: parameters for generating synthetic datasets
    """
    weights_c = np.array([.5, .5])  # mixture weights
    means_c = 3 * np.array([-1 * np.ones((d_c, 1)), np.ones((d_c, 1))])  # mixture means
    m = weights_c.shape[0]  # number of components in MoG
    covs_c = np.zeros((m, d_c, d_c))
    for i in range(m):
        covs_c[i] = .5 * np.eye(d_c)  # mixture (co)variances

    a_y = .5 * np.ones((d_c, 1))  # strength of influence of x_c
    b_y = 0 * np.ones(1)  # class boundary

    a_e0 = 0.5 * np.ones((d_c, d_e))  # dependence of x_e on x_c for class y=0
    a_e1 = -0.5 * np.ones((d_c, d_e))  # dependence of x_e on x_c for class y=0
    mu_y = 0  # dependence of x_e on y
    b_0 = -mu_y * np.ones((1, d_e))
    b_1 = mu_y * np.ones((1, d_e))
    cov_e0 = .5 * np.eye(d_e)  # noise variance for n_e
    cov_e1 = .5 * np.eye(d_e)  # noise variance for n_e
    return weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1


# Get parameters
n_labelled = 10
n_unlabelled = 200
n_iterations = 100
d_c = 2  # input dimension
d_e = 2  # output dimension
weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = get_params(d_c, d_e)


# Plot Data for 1D case
if d_c == 1 and d_e == 1:
    # Get data
    x_c, y, x_e = get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                  a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
    z_c, z_y, z_e = get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                    a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_unlabelled)
    fig_data = plot_data(x_c, y, x_e, z_c, z_e)
    plt.show()


# Initialise result matrices
acc_lin_lr = []
acc_lin_tsvm = []
acc_rbf_tsvm = []
acc_rbf_label_prop = []
acc_rbf_label_spread = []
acc_knn_label_prop = []
acc_knn_label_spread = []
acc_semigen_labelled = []
acc_soft_EM = []
acc_hard_EM = []
acc_cond_prop = []


# Simulate
for i in range(n_iterations):
    print 'iteration: ', i, '/', n_iterations
    x_c, y, x_e = get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                      a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
    z_c, z_y, z_e = get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                        a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_unlabelled)

    a_lin_lr, a_lin_tsvm, a_rbf_tsvm, a_rbf_label_prop, a_rbf_label_spread, a_knn_label_prop, \
        a_knn_label_spread, a_semigen_labelled, a_soft_EM, a_hard_EM, a_cond_prop\
        = ssl.run_methods(x_c, y, x_e, z_c, z_y, z_e)

    # Store results
    acc_lin_lr.append(a_lin_lr)
    acc_lin_tsvm.append(a_lin_tsvm)
    acc_rbf_tsvm.append(a_rbf_tsvm)
    acc_rbf_label_prop.append(a_rbf_label_prop)
    acc_rbf_label_spread.append(a_rbf_label_spread)
    acc_knn_label_prop.append(a_knn_label_prop)
    acc_knn_label_spread.append(a_knn_label_spread)
    acc_semigen_labelled.append(a_semigen_labelled)
    acc_soft_EM.append(a_soft_EM)
    acc_hard_EM.append(a_hard_EM)
    acc_cond_prop.append(a_cond_prop)

print 'Accuracy of linear logistic regression: ', np.mean(acc_lin_lr), ' +/- ', np.std(acc_lin_lr)
print 'Accuracy of linear TSVM: ', np.mean(acc_lin_tsvm), ' +/- ', np.std(acc_lin_tsvm)
print 'Accuracy of rbf TSVM: ', np.mean(acc_rbf_tsvm), ' +/- ', np.std(acc_rbf_tsvm)
print 'Accuracy of semi-gen model (labelled only): ', np.mean(acc_semigen_labelled), ' +/- ', np.std(acc_semigen_labelled)
print 'Accuracy of soft EM: ', np.mean(acc_soft_EM), ' +/- ', np.std(acc_soft_EM)
print 'Accuracy of hard EM: ', np.mean(acc_hard_EM), ' +/- ', np.std(acc_hard_EM)
print 'Accuracy of conditional prop: ', np.mean(acc_cond_prop), ' +/- ', np.std(acc_cond_prop)
print 'Accuracy of rbf label spread: ', np.mean(acc_rbf_label_spread), ' +/- ', np.std(acc_rbf_label_spread)
print 'Accuracy of rbf label prop: ', np.mean(acc_rbf_label_prop), ' +/- ', np.std(acc_rbf_label_prop)
print 'Accuracy of knn label prop: ', np.mean(acc_knn_label_prop), ' +/- ', np.std(acc_knn_label_prop)
print 'Accuracy of knn label spread: ', np.mean(acc_knn_label_spread), ' +/- ', np.std(acc_knn_label_spread)