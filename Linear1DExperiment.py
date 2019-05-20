import SSLCauseEffect as ssl
import numpy as np

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
# import time
# start = time.time()
# end = time.time()
# print(end - start)


# Define parameters
n_labelled = 10
n_unlabelled = 200
n_iterations = 100
d_c = 1  # input dimension
d_e = 1  # output dimension
weights_c, means_c, covs_c, a_y, b_y, a_e0, a_e1, b_0, b_1, cov_e0, cov_e1 = ssl.get_params(d_c, d_e)

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
acc_semigen_labelled = []
acc_soft_EM = []
acc_hard_EM = []
for i in range(n_iterations):
    x_c, y, x_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                      a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_labelled)
    z_c, z_y, z_e = ssl.get_data_linear(weights_c, means_c, covs_c, a_y, b_y,
                                        a_e0, a_e1, b_0, b_1, cov_e0, cov_e1, n_unlabelled)
    acc_lin_lr, acc_lin_tsvm, acc_rbf_tsvm, acc_rbf_label_prop, acc_rbf_label_spread, acc_knn_label_prop, \
        acc_knn_label_spread, acc_semigen_labelled, acc_soft_EM, acc_hard_EM \
        = ssl.collect_results(acc_lin_lr, acc_lin_tsvm, acc_rbf_tsvm, acc_rbf_label_prop, acc_rbf_label_spread,
                              acc_knn_label_prop, acc_knn_label_spread, acc_semigen_labelled, acc_soft_EM, acc_hard_EM,
                              x_c, y, x_e, z_c, z_y, z_e)

print 'Accuracy of linear logistic regression: ', np.mean(acc_lin_lr), ' +/- ', np.std(acc_lin_lr)
print 'Accuracy of linear TSVM: ', np.mean(acc_lin_tsvm), ' +/- ', np.std(acc_lin_tsvm)
print 'Accuracy of rbf TSVM: ', np.mean(acc_rbf_tsvm), ' +/- ', np.std(acc_rbf_tsvm)
print 'Accuracy of semi-gen model (labelled only): ', np.mean(acc_semigen_labelled), ' +/- ', np.std(acc_semigen_labelled)
print 'Accuracy of soft EM: ', np.mean(acc_soft_EM), ' +/- ', np.std(acc_soft_EM)
print 'Accuracy of hard EM: ', np.mean(acc_hard_EM), ' +/- ', np.std(acc_hard_EM)
print 'Accuracy of rbf label prop: ', np.mean(acc_rbf_label_prop), ' +/- ', np.std(acc_rbf_label_prop)
print 'Accuracy of rbf label spread: ', np.mean(acc_rbf_label_spread), ' +/- ', np.std(acc_rbf_label_spread)
print 'Accuracy of knn label prop: ', np.mean(acc_knn_label_prop), ' +/- ', np.std(acc_knn_label_prop)
print 'Accuracy of knn label spread: ', np.mean(acc_knn_label_spread), ' +/- ', np.std(acc_knn_label_spread)
