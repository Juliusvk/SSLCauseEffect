import pandas as pd
import numpy as np
from numpy.random import shuffle
import SSLCauseEffect as ssl
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# Set simulation settings
n_iter = 100
n_labelled = 20
n_unlabelled = 200

# Read data
heart = pd.read_csv("heart.csv")
data = heart.values.astype(float)
idx = np.arange(data.shape[0])

# Define cause and effect features and target
idx_cau = [11, 1, 12]
idx_eff = [2] # 2 (cp) is categorical effect feature
idx_y = [13]

# standardise integer feature 11 (ca: no. or vessels colored by fluoroscopy)
data[:, 11] = np.divide(data[:, 11] - np.mean(data[:, 11]), np.std(data[:, 11]))

# dummy code categorical feature 12 (thal: thallium scintigraphy results)
dummy_thal = pd.get_dummies(data[:, 12]).values.astype(float)
idx_cau = np.setdiff1d(idx_cau, 12)
data_c = np.concatenate((data[:, idx_cau], dummy_thal[:, 1:]), axis=1)

data_e = data[:, idx_eff]


# Initialise result arrays
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


for i in range(n_iter):
    print 'iteration: ', i, '/', n_iter
    # draw random sample ensuring 2 observations per class
    n_0 = 0
    n_1 = 0
    while n_0 < 2 or n_1 < 2:
        shuffle(idx)
        y = data[idx[0:n_labelled], idx_y]
        y = np.reshape(y, (y.shape[0], 1))
        n_0 = sum(y == 0)
        n_1 = sum(y == 1)

    x_c = data_c[idx[0:n_labelled]]
    x_e = data_e[idx[0:n_labelled]]
    z_c = data_c[idx[n_labelled:n_labelled + n_unlabelled]]
    z_e = data_e[idx[n_labelled:n_labelled + n_unlabelled]]
    z_y = data[idx[n_labelled:n_labelled+n_unlabelled], idx_y]
    z_y = np.reshape(z_y, (z_y.shape[0], 1))

    a_lin_lr, a_lin_tsvm, a_rbf_tsvm, a_rbf_label_prop, a_rbf_label_spread, a_knn_label_prop, \
    a_knn_label_spread, a_semigen_labelled, a_soft_EM, a_hard_EM, a_cond_prop \
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
print 'Accuracy of semi-gen model (labelled only): ', np.mean(acc_semigen_labelled), ' +/- ', np.std(
    acc_semigen_labelled)
print 'Accuracy of soft EM: ', np.mean(acc_soft_EM), ' +/- ', np.std(acc_soft_EM)
print 'Accuracy of hard EM: ', np.mean(acc_hard_EM), ' +/- ', np.std(acc_hard_EM)
print 'Accuracy of cond prop: ', np.mean(acc_cond_prop), ' +/- ', np.std(acc_cond_prop)
print 'Accuracy of rbf label spread: ', np.mean(acc_rbf_label_spread), ' +/- ', np.std(acc_rbf_label_spread)
print 'Accuracy of knn label spread: ', np.mean(acc_knn_label_spread), ' +/- ', np.std(acc_knn_label_spread)
print 'Accuracy of rbf label prop: ', np.mean(acc_rbf_label_prop), ' +/- ', np.std(acc_rbf_label_prop)
# print 'Accuracy of knn label prop: ', np.mean(acc_knn_label_prop), ' +/- ', np.std(acc_knn_label_prop)
