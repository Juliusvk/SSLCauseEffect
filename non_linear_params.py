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