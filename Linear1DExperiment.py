import SSLCauseEffect as ssl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Define parameters
m = 3  # number of components in MoG
d_c = 1  # input dimension
d_e = 1  # output dimension

weights_c = np.array([.3, .4, .3])  # mixture weights
means_c = 2 * np.array([[-1], [0], [1]])  # mixture means
covs_c = np.zeros((m, d_c, d_c))
for i in range(m):
    covs_c[i] = 0.1 * np.eye(d_c)  # mixture (co)variances

a_y = 1 * np.ones((d_c, 1))  # strength of influence of x_c
b_y = np.zeros((1, 1))  # class boundary

a_e = 0 * np.ones((d_c, d_e))  # dependence of x_e on x_c
mu_y = 2  # dependence of x_e on y
b_0 = -mu_y * np.ones((1, d_e))
b_1 = mu_y * np.ones((1, d_e))
cov_e = np.eye(d_e)  # noise variance for n_e


# Generate Data
n_labelled = 50
n_unlabelled = 500
x_c, y, x_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y, a_e, b_0, b_1, cov_e, n_labelled)
z_c, z_y, z_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y, a_e, b_0, b_1, cov_e, n_unlabelled)


# Plot Data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(z_c, z_e, color='grey', marker='.')
ax.scatter(x_c[y == 0], x_e[y == 0], color='blue', marker='.')
ax.scatter(x_c[y == 1], x_e[y == 1], color='red', marker='.')
ax.set(xlabel='Causal features $X_C$', ylabel='Effect features $X_E$')
# ax.legend(loc='best')
# plt.show()

py1 = ssl.sigmoid(ssl.fy_linear(x_c, a_y, b_y))  # P(Y=1 |X_C)
mean1 = np.matmul(x_c, a_e) + b_1
pe1 = np.zeros(py1.shape)
for i in range(py1.shape[0]):
    pe1[i] = multivariate_normal.pdf(x_e[i], mean1[i], cov_e)  # P(X_E| X_C, Y=1)
