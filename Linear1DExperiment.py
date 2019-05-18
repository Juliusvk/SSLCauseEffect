import SSLCauseEffect as ssl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from scikitTSVM import SKTSVM
import random as rnd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
# import time
# start = time.time()
# end = time.time()
# print(end - start)

# Define parameters
d_c = 3  # input dimension
d_e = 2  # output dimension
weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = ssl.get_params(d_c, d_e)


# Generate Data
n_labelled = 10
n_unlabelled = 15
x_c, y, x_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
z_c, z_y, z_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_unlabelled)
x = np.concatenate((x_c, x_e), axis=1)
z = np.concatenate((z_c, z_e), axis=1)


# Plot Data
if d_c == 1 and d_e == 1:
    fig_data = ssl.plot_data(x_c, y, x_e, z_c, z_e)
    plt.show()

soft_label_true = ssl.predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)


# Baseline: Linear Logistic Regression
lin_lr = LogisticRegression(random_state=0, solver='liblinear').fit(x, y.ravel())
hard_label_lin_lr = lin_lr.predict(z)
soft_label_lin_lr = lin_lr.predict_proba(z)[:, 1]
acc_lin_lr = lin_lr.score(z, z_y)
print 'Accuracy of linear logistic regression: ', acc_lin_lr

# TRANSDUCTIVE APPROACHES
# merge labelled and unlabelled data (with label -1) for transductive methods
x_merged = np.concatenate((x, z))
y_merged = np.concatenate((y, -1*np.ones((z.shape[0], 1)))).ravel().astype(int)

# Baseline: Linear TSVM: https://github.com/tmadl/semisup-learn/tree/master/methods
lin_tsvm = SKTSVM(kernel='linear')
lin_tsvm.fit(x_merged, y_merged)
# hard_label_lin_tsvm = lin_tsvm.predict(z)
# soft_label_lin_tsvm = lin_tsvm.predict_proba(z)[:, 1]
acc_lin_tsvm = lin_tsvm.score(z, z_y)
print 'Accuracy of linear TSVM: ', acc_lin_tsvm

# Baseline: Non-Linear TSVM:  https://github.com/tmadl/semisup-learn/tree/master/methods
rbf_tsvm = SKTSVM(kernel='RBF')
rbf_tsvm.fit(x_merged, y_merged)
# hard_label_rbf_tsvm = rbf_tsvm.predict(z)
# soft_label_rbf_tsvm = rbf_tsvm.predict_proba(z)[:, 1]
acc_rbf_tsvm = rbf_tsvm.score(z, z_y)
print 'Accuracy of rbf TSVM: ', acc_rbf_tsvm

# # Baseline: Label Propagation RBF weights
# rbf_label_prop = LabelPropagation(kernel='rbf', gamma=20)
# rbf_label_prop.fit(x_merged, y_merged)
# # hard_label_rbf_label_prop= rbf_label_prop.predict(z)
# # soft_label_rbf_label_prop = rbf_label_prop.predict_proba(z)[:, 1]
# acc_rbf_label_prop = rbf_label_prop.score(z, z_y)
# print 'Accuracy of rbf label prop: ', acc_rbf_label_prop
#
# # Baseline: Label Spreading with RBF weights
# rbf_label_spread = LabelSpreading(kernel='rbf', gamma=20)
# rbf_label_spread.fit(x_merged, y_merged)
# # hard_label_rbf_label_spread = rbf_label_spread.predict(z)
# # soft_label_rbf_label_spread = rbf_label_spread.predict_proba(z)[:, 1]
# acc_rbf_label_spread = rbf_label_spread.score(z, z_y)
# print 'Accuracy of rbf label spread: ', acc_rbf_label_spread
#
# # THE K-NN VERSIONS ARE UNSTABLE UNLESS USING LARGE K
# # Baseline: Label Propagation with k-NN weights
# knn_label_prop = LabelPropagation(kernel='knn', n_neighbors=11)
# knn_label_prop.fit(x_merged, y_merged)
# # hard_label_knn_label_prop = knn_label_prop.predict(z)
# # soft_label_knn_label_prop = knn_label_prop.predict_proba(z)[:, 1]
# acc_knn_label_prop = knn_label_prop.score(z, z_y)
# print 'Accuracy of knn label prop: ', acc_knn_label_prop
#
# # Baseline: Label Spreading with k-NN weights
# knn_label_spread = LabelSpreading(kernel='knn', n_neighbors=11)
# knn_label_spread.fit(x_merged, y_merged)
# # hard_label_knn_label_spread = knn_label_spread.predict(z)
# # soft_label_knn_label_spread = knn_label_spread.predict_proba(z)[:, 1]
# acc_knn_label_spread = knn_label_spread.score(z, z_y)
# print 'Accuracy of knn label spread: ', acc_knn_label_spread


# INDUCTIVE METHODS
# Generative Models
# Semi-generative model on labelled data only

# Semi-generative model on all data using EM (with hard labels)
a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = ssl.hard_label_EM(x_c, y, x_e, z_c, z_e)


a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = ssl.soft_label_EM(x_c, y, x_e, z_c, z_e)

# M-step
# lr_c_only = LogisticRegression(random_state=0).fit(x_c_merged[y_merged != -1], y_merged[y_merged != -1].ravel())
# a_y = np.transpose(lr_c_only.coef_)
# b_y = lr_c_only.intercept_
#
# # unweighted regression
# ridge_e0 = Ridge().fit(x_c_merged[y_merged == 0], x_e_merged[y_merged == 0])
# a_e0 = ridge_e0.coef_
# b_e0 = ridge_e0.intercept_.reshape((1, d_e))
# sq_res_0 = np.square(x_e_merged[y_merged == 0] - ridge_e0.predict(x_c_merged[y_merged == 0]))
# cov_e0 = np.diag(np.divide(np.sum(sq_res_0, axis=0), x_e_merged[y_merged == 0].shape[0]))
#
# ridge_e1 = Ridge().fit(x_c_merged[y_merged == 1], x_e_merged[y_merged == 1])
# a_e1 = ridge_e1.coef_
# b_e1 = ridge_e1.intercept_.reshape((1, d_e))
# sq_res_1 = np.square(x_e_merged[y_merged == 1] - ridge_e1.predict(x_c_merged[y_merged == 1]))
# cov_e1 = np.diag(np.divide(np.sum(sq_res_1, axis=0), x_e_merged[y_merged == 1].shape[0]))
#
# # weighted regression
# ridge_e0 = Ridge().fit(x_c_merged[y_merged != -1], x_e_merged[y_merged != -1], sample_weight=1-y_merged[y_merged != -1])
# alpha_e0 = ridge_e0.coef_
# beta_e0 = ridge_e0.intercept_.reshape((1, d_e))
# sq_res_0 = np.square(x_e_merged[y_merged != -1] - ridge_e0.predict(x_c_merged[y_merged != -1]))
# cov_e0 = np.diag(np.divide(np.sum(np.multiply((1-y_merged[y_merged != -1]).reshape((y_merged[y_merged != -1].shape[0], 1)), sq_res_0), axis=0), np.sum(1-y_merged[y_merged != -1])))
#
# ridge_e1 = Ridge().fit(x_c_merged[y_merged != -1], x_e_merged[y_merged != -1], sample_weight=y_merged[y_merged != -1])
# alpha_e1 = ridge_e1.coef_
# beta_e1 = ridge_e1.intercept_.reshape((1, d_e))
# sq_res_1 = np.square(x_e_merged[y_merged != -1] - ridge_e1.predict(x_c_merged[y_merged != -1]))
# cov_e1 = np.diag(np.divide(np.sum(np.multiply((y_merged[y_merged != -1]).reshape((y_merged[y_merged != -1].shape[0], 1)), sq_res_1), axis=0), np.sum(y_merged[y_merged != -1])))


