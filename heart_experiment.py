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

# dummy code categorical effect feature 2: (cp: chest pain type)
# dummy_cp = pd.get_dummies(data[:, 2]).values.astype(float)
# data_e = dummy_cp[:, :-1]
data_e = data[:, idx_eff]
classes = np.unique(data_e.astype(int))

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
acc_disc_eff_hard = []
acc_disc_eff_soft = []
acc_disc_eff_semigen = []
acc_disc_eff_cond_prop = []

for i in range(n_iter):
    print 'iteration: ', i, '/', n_iter
    # ensure all values of effect feature have been seen for each class at least once
    e_0 = False
    e_1 = False
    while not e_0 or not e_1:
        shuffle(idx)
        y = data[idx[0:n_labelled], idx_y].reshape((-1, 1))
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        x_e = data_e[idx[0:n_labelled]]
        x_e0 = x_e[idx_0].astype(int)
        x_e1 = x_e[idx_1].astype(int)
        e_0 = all(elem in x_e0 for elem in classes)
        e_1 = all(elem in x_e1 for elem in classes)

    x_c = data_c[idx[0:n_labelled]]
    z_c = data_c[idx[n_labelled:n_labelled + n_unlabelled]]
    z_e = data_e[idx[n_labelled:n_labelled + n_unlabelled]]
    z_y = data[idx[n_labelled:n_labelled + n_unlabelled], idx_y].reshape((-1, 1))


    # Get results
    acc_disc_eff_hard.append(ssl.disc_eff_hard_EM(x_c, y, x_e, z_c, z_y, z_e))
    acc_disc_eff_soft.append(ssl.disc_eff_soft_EM(x_c, y, x_e, z_c, z_y, z_e))
    acc_disc_eff_semigen_temp = np.mean((ssl.disc_eff_semigen(x_c, y, x_e, z_c, z_e) > 0.5) == z_y)
    acc_disc_eff_semigen.append(acc_disc_eff_semigen_temp)
    acc_disc_eff_cond_prop.append(ssl.disc_cond_prop(x_c, y, x_e, z_c, z_y, z_e))

    a_lin_lr, a_lin_tsvm, a_rbf_tsvm, a_rbf_label_prop, a_rbf_label_spread, a_knn_label_prop, \
    a_knn_label_spread, a_semigen_labelled, a_soft_EM, a_hard_EM, a_cond_prop \
        = ssl.run_methods(x_c, y, x_e, z_c, z_y, z_e)

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
print 'Accuracy of semi-gen model (labelled only) - DISCRETE: ', np.mean(acc_disc_eff_semigen), ' +/- ', np.std(
    acc_disc_eff_semigen)
print 'Accuracy of soft EM - DISCRETE: ', np.mean(acc_disc_eff_soft), ' +/- ', np.std(acc_disc_eff_soft)
print 'Accuracy of hard EM - DISCRETE: ', np.mean(acc_disc_eff_hard), ' +/- ', np.std(acc_disc_eff_hard)
print 'Accuracy of cond prop - DISCRETE: ', np.mean(acc_disc_eff_cond_prop), ' +/- ', np.std(acc_disc_eff_cond_prop)
print 'Accuracy of rbf label spread: ', np.mean(acc_rbf_label_spread), ' +/- ', np.std(acc_rbf_label_spread)
print 'Accuracy of knn label spread: ', np.mean(acc_knn_label_spread), ' +/- ', np.std(acc_knn_label_spread)
print 'Accuracy of rbf label prop: ', np.mean(acc_rbf_label_prop), ' +/- ', np.std(acc_rbf_label_prop)
# print 'Accuracy of knn label prop: ', np.mean(acc_knn_label_prop), ' +/- ', np.std(acc_knn_label_prop)
