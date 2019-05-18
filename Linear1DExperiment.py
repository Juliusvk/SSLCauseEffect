import SSLCauseEffect as ssl
import numpy as np
from sklearn.linear_model import LogisticRegression
from scikitTSVM import SKTSVM
import random as rnd
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt

# import time
# start = time.time()
# end = time.time()
# print(end - start)

# Define parameters
d_c = 1  # input dimension
d_e = 1  # output dimension

weights_c = np.array([.5, .5])  # mixture weights
means_c = 1 * np.array([[-1], [1]])  # mixture means
m = weights_c.shape[0] # number of components in MoG
covs_c = np.zeros((m, d_c, d_c))
for i in range(m):
    covs_c[i] = 0.1 * np.eye(d_c)  # mixture (co)variances

a_y = 1 * np.ones((d_c, 1))  # strength of influence of x_c
b_y = np.zeros((1, 1))  # class boundary

a_e0 = 3 * np.ones((d_c, d_e))  # dependence of x_e on x_c for class y=0
a_e1 = -2 * np.ones((d_c, d_e)) # dependence of x_e on x_c for class y=0
mu_y = 0  # dependence of x_e on y
b_0 = -mu_y * np.ones((1, d_e))
b_1 = mu_y * np.ones((1, d_e))
cov_e0 = 1 * np.eye(d_e)  # noise variance for n_e
cov_e1 = 1 * np.eye(d_e)  # noise variance for n_e

# Generate Data
n_labelled = 8
n_unlabelled = 256
x_c, y, x_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
z_c, z_y, z_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_unlabelled)

# Plot Data
fig_data = ssl.plot_data(x_c, y, x_e, z_c, z_e)
plt.show()

soft_label_true = ssl.predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)

x = np.concatenate((x_c, x_e), axis=1)
z = np.concatenate((z_c, z_e), axis=1)
x_merged = np.concatenate((x, z))
y_merged = np.concatenate((y, -1*np.ones((z.shape[0], 1)))).ravel().astype(int) #-1 for unlabelled data

# Baseline: Linear Logistic Regression
lin_lr = LogisticRegression(random_state=0).fit(x, y.ravel())
hard_label_lin_lr = lin_lr.predict(z)
soft_label_lin_lr = lin_lr.predict_proba(z)[:, 1]
acc_lin_lr = lin_lr.score(z, z_y)
print 'Accuracy of linear logistic regression: ', acc_lin_lr


# Baseline: Linear TSVM: https://github.com/tmadl/semisup-learn/tree/master/methods
lin_tsvm = SKTSVM(kernel='linear')
lin_tsvm.fit(x_merged, y_merged)
hard_label_lin_tsvm = lin_tsvm.predict(z)
soft_label_lin_tsvm = lin_tsvm.predict_proba(z)[:, 1]
acc_lin_tsvm = lin_tsvm.score(z, z_y)
print 'Accuracy of linear TSVM: ', acc_lin_tsvm


# Baseline: Non-Linear TSVM:  https://github.com/tmadl/semisup-learn/tree/master/methods
rbf_tsvm = SKTSVM(kernel='RBF')
rbf_tsvm.fit(x_merged, y_merged)
hard_label_rbf_tsvm = rbf_tsvm.predict(z)
soft_label_rbf_tsvm = rbf_tsvm.predict_proba(z)[:, 1]
acc_rbf_tsvm = rbf_tsvm.score(z, z_y)
print 'Accuracy of rbf TSVM: ', acc_rbf_tsvm

# Baseline: Non-Linear Logistic Regression


# Baseline: Label Propagation RBF weights

# Baseline: Label Propagation k-NN weights


