import SSLCauseEffect as ssl
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# Define parameters
n_labelled = 10
n_unlabelled = 200
n_iterations = 100
d_c = 2  # input dimension
d_e = 2  # output dimension


def get_params(d_c, d_e):
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
weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = get_params(d_c, d_e)

# Get data
x_c, y, x_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                  a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
z_c, z_y, z_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                    a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_unlabelled)

# Plot Data
if d_c == 1 and d_e == 1:
    fig_data = ssl.plot_data(x_c, y, x_e, z_c, z_e)
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
    x_c, y, x_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                      a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
    z_c, z_y, z_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
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
# print acc_rbf_label_prop
# print acc_rbf_label_spread
# print acc_knn_label_prop
# print acc_knn_label_prop