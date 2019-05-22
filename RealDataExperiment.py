import pandas as pd
import numpy as np
from numpy.random import shuffle
import SSLCauseEffect as ssl
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

n_iter = 100
n_labelled = 20
n_unlabelled = 200

diabetes = pd.read_csv("diabetes.csv")
data = diabetes.values.astype(float)
idx = np.arange(data.shape[0])
idx_cau = [1]
idx_eff = [5]

# heart = pd.read_csv("heart.csv")
# data = heart.values.astype(float)
# idx = np.arange(data.shape[0])
# idx_cau = [11, 1, 3]
# idx_eff = [2, 12, 9, 8, 7]
#pd.get_dummies(data.cp)

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
for i in range(n_iter):
    print 'iteration: ', i, '/', n_iter
    n_0 = 0
    n_1 = 0
    while n_0 < 2 or n_1 < 2:
        shuffle(idx)
        data_l = data[idx[0:n_labelled]]
        data_u = data[idx[n_labelled:n_labelled+n_unlabelled]]
        x_c = data_l[:, idx_cau]
        x_e = data_l[:, idx_eff]
        y = np.reshape(data_l[:, 8], (data_l[:, 8].shape[0], 1))
        n_0 = sum(y == 0)
        n_1 = sum(y == 1)

    z_c = data_u[:, idx_cau]
    z_e = data_u[:, idx_eff]
    z_y = np.reshape(data_u[:, 8], (data_u[:, 8].shape[0], 1))
    acc_lin_lr, acc_lin_tsvm, acc_rbf_tsvm, acc_rbf_label_prop, acc_rbf_label_spread, acc_knn_label_prop, \
        acc_knn_label_spread, acc_semigen_labelled, acc_soft_EM, acc_hard_EM \
        = ssl.collect_results(acc_lin_lr, acc_lin_tsvm, acc_rbf_tsvm, acc_rbf_label_prop, acc_rbf_label_spread,
                              acc_knn_label_prop, acc_knn_label_spread, acc_semigen_labelled, acc_soft_EM, acc_hard_EM,
                              x_c, y, x_e, z_c, z_y, z_e)

print 'Accuracy of linear logistic regression: ', np.mean(acc_lin_lr), ' +/- ', np.std(acc_lin_lr)
print 'Accuracy of linear TSVM: ', np.mean(acc_lin_tsvm), ' +/- ', np.std(acc_lin_tsvm)
print 'Accuracy of rbf TSVM: ', np.mean(acc_rbf_tsvm), ' +/- ', np.std(acc_rbf_tsvm)
print 'Accuracy of semi-gen model (labelled only): ', np.mean(acc_semigen_labelled), ' +/- ', np.std(
    acc_semigen_labelled)
print 'Accuracy of soft EM: ', np.mean(acc_soft_EM), ' +/- ', np.std(acc_soft_EM)
print 'Accuracy of hard EM: ', np.mean(acc_hard_EM), ' +/- ', np.std(acc_hard_EM)
print 'Accuracy of rbf label spread: ', np.mean(acc_rbf_label_spread), ' +/- ', np.std(acc_rbf_label_spread)
print 'Accuracy of knn label spread: ', np.mean(acc_knn_label_spread), ' +/- ', np.std(acc_knn_label_spread)
print 'Accuracy of rbf label prop: ', np.mean(acc_rbf_label_prop), ' +/- ', np.std(acc_rbf_label_prop)
# print 'Accuracy of knn label prop: ', np.mean(acc_knn_label_prop), ' +/- ', np.std(acc_knn_label_prop)


# heart = pd.read_csv("heart.csv")
# # print heart.head()
# X_C = np.transpose(np.array([heart.ca, heart.sex, heart.trestbps, heart.chol]))
# X_E = np.transpose(np.array([heart.cp, heart.thal, heart.oldpeak, heart.exang]))
# Y = np.transpose(np.array([heart.target]))
# print diabetes.head()
# X_C = np.transpose(np.array([diabetes.BMI, diabetes.Pregnancies, diabetes.DiabetesPedigreeFunction, diabetes.Age]))
# X_E = np.transpose(np.array([diabetes.Glucose, diabetes.BloodPressure, diabetes.Insulin]))
# Y = np.transpose(np.array([diabetes.Outcome]))
