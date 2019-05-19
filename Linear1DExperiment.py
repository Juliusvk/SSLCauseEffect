import SSLCauseEffect as ssl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from scikitTSVM import SKTSVM
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
# import time
# start = time.time()
# end = time.time()
# print(end - start)


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
    cov_e0 = .25 * np.eye(d_e)  # noise variance for n_e
    cov_e1 = .25 * np.eye(d_e)  # noise variance for n_e
    return weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1


# Define parameters
d_c = 1  # input dimension
d_e = 1  # output dimension
weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = get_params(d_c, d_e)

# Generate Data
n_labelled = 16
n_unlabelled = 256
n_iterations = 10

x_c, y, x_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                  a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
z_c, z_y, z_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                    a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_unlabelled)
# soft_label_true = ssl.predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)

# Plot Data
if d_c == 1 and d_e == 1:
    fig_data = ssl.plot_data(x_c, y, x_e, z_c, z_e)
    plt.show()

acc_lin_lr = []
acc_lin_tsvm = []
acc_rbf_tsvm = []
acc_rbf_label_prop = []
acc_rbf_label_spread = []
acc_knn_label_prop = []
acc_knn_label_spread = []
acc_soft_EM = []
acc_hard_EM = []

for i in range(n_iterations):
    x_c, y, x_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                      a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
    z_c, z_y, z_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                        a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_unlabelled)
    x = np.concatenate((x_c, x_e), axis=1)
    z = np.concatenate((z_c, z_e), axis=1)

    # Baseline: Linear Logistic Regression
    lin_lr = LogisticRegression(random_state=0, solver='liblinear').fit(x, y.ravel())
    acc_lin_lr.append(lin_lr.score(z, z_y))
    # hard_label_lin_lr = lin_lr.predict(z)
    # soft_label_lin_lr = lin_lr.predict_proba(z)[:, 1]

    # TRANSDUCTIVE APPROACHES
    # merge labelled and unlabelled data (with label -1) for transductive methods
    x_merged = np.concatenate((x, z))
    y_merged = np.concatenate((y, -1*np.ones((z.shape[0], 1)))).ravel().astype(int)

    # Baseline: Linear TSVM: https://github.com/tmadl/semisup-learn/tree/master/methods
    lin_tsvm = SKTSVM(kernel='linear')
    lin_tsvm.fit(x_merged, y_merged)
    acc_lin_tsvm.append(lin_tsvm.score(z, z_y))
    # hard_label_lin_tsvm = lin_tsvm.predict(z)
    # soft_label_lin_tsvm = lin_tsvm.predict_proba(z)[:, 1]

    # Baseline: Non-Linear TSVM:  https://github.com/tmadl/semisup-learn/tree/master/methods
    rbf_tsvm = SKTSVM(kernel='RBF')
    rbf_tsvm.fit(x_merged, y_merged)
    acc_rbf_tsvm.append(rbf_tsvm.score(z, z_y))
    # hard_label_rbf_tsvm = rbf_tsvm.predict(z)
    # soft_label_rbf_tsvm = rbf_tsvm.predict_proba(z)[:, 1]

    # Baseline: Label Propagation RBF weights
    try:
        rbf_label_prop = LabelPropagation(kernel='rbf', gamma=20)
        rbf_label_prop.fit(x_merged, y_merged)
        acc_rbf_label_prop.append(rbf_label_prop.score(z, z_y))
        # hard_label_rbf_label_prop= rbf_label_prop.predict(z)
        # soft_label_rbf_label_prop = rbf_label_prop.predict_proba(z)[:, 1]
    except:
        print 'rbf label prop did not work'

    # Baseline: Label Spreading with RBF weights
    try:
        rbf_label_spread = LabelSpreading(kernel='rbf', gamma=20)
        rbf_label_spread.fit(x_merged, y_merged)
        acc_rbf_label_spread.append(rbf_label_spread.score(z, z_y))
        # hard_label_rbf_label_spread = rbf_label_spread.predict(z)
        # soft_label_rbf_label_spread = rbf_label_spread.predict_proba(z)[:, 1]
    except:
        print 'rbf label spread did not work '

    # THE K-NN VERSIONS ARE UNSTABLE UNLESS USING LARGE K
    # Baseline: Label Propagation with k-NN weights
    try:
        knn_label_prop = LabelPropagation(kernel='knn', n_neighbors=11)
        knn_label_prop.fit(x_merged, y_merged)
        acc_knn_label_prop.append(knn_label_prop.score(z, z_y))
        # hard_label_knn_label_prop = knn_label_prop.predict(z)
        # soft_label_knn_label_prop = knn_label_prop.predict_proba(z)[:, 1]
    except:
        print 'knn label prop did not work'

    # Baseline: Label Spreading with k-NN weights
    try:
        knn_label_spread = LabelSpreading(kernel='knn', n_neighbors=11)
        knn_label_spread.fit(x_merged, y_merged)
        acc_knn_label_spread.append(knn_label_spread.score(z, z_y))
        # hard_label_knn_label_spread = knn_label_spread.predict(z)
        # soft_label_knn_label_spread = knn_label_spread.predict_proba(z)[:, 1]
    except:
        print 'knn label spread did not work'

    # Generative Models
    # Semi-generative model on labelled data only
    # EM with soft labels
    a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = ssl.soft_label_EM(x_c, y, x_e, z_c, z_e)
    soft_label_soft_EM = ssl.predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)
    hard_label_soft_EM = soft_label_soft_EM > 0.5
    acc_soft_EM.append(np.mean(hard_label_soft_EM == z_y))

    # EM with hard labels
    a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = ssl.hard_label_EM(x_c, y, x_e, z_c, z_e)
    soft_label_hard_EM = ssl.predict_class_probs(z_c, z_e, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1)
    hard_label_hard_EM = soft_label_hard_EM > 0.5
    acc_hard_EM.append(np.mean(hard_label_hard_EM == z_y))


print 'Accuracy of linear logistic regression: ', np.mean(acc_lin_lr), ' +/- ', np.std(acc_lin_lr)
print 'Accuracy of linear TSVM: ', np.mean(acc_lin_tsvm), ' +/- ', np.std(acc_lin_tsvm)
print 'Accuracy of rbf TSVM: ', np.mean(acc_rbf_tsvm), ' +/- ', np.std(acc_rbf_tsvm)
print 'Accuracy of soft EM: ', np.mean(acc_soft_EM), ' +/- ', np.std(acc_soft_EM)
print 'Accuracy of hard EM: ', np.mean(acc_hard_EM), ' +/- ', np.std(acc_hard_EM)
print 'Accuracy of rbf label prop: ', np.mean(acc_rbf_label_prop), ' +/- ', np.std(acc_rbf_label_prop)
print 'Accuracy of rbf label spread: ', np.mean(acc_rbf_label_spread), ' +/- ', np.std(acc_rbf_label_spread)
# print 'Accuracy of knn label prop: ', np.mean(acc_knn_label_prop), ' +/- ', np.std(acc_knn_label_prop)
print 'Accuracy of knn label spread: ', np.mean(acc_knn_label_spread), ' +/- ', np.std(acc_knn_label_spread)
