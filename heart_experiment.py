from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from scikitTSVM import SKTSVM
import pandas as pd

heart = pd.read_csv("heart.csv")
heart_data = heart.values.astype(float)
idx = np.arange(heart_data.shape[0])
n_iter = 10
n_labelled = 20
n_unlabelled = 200
idx_cau = [11, 1, 3]
idx_eff = [2, 7, 8, 9, 12]


def discrete_data_EM(x_c, y, x_e, z_c, z_y, z_e):
    c = np.concatenate((x_c, z_c))
    e = np.concatenate((x_e, z_e))
    LRC = LogisticRegression(random_state=0, solver='liblinear')
    LRC.fit(x_c, y.ravel())
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    LR0 = LogisticRegression(random_state=0, solver='liblinear')
    LR0.fit(x_c[idx_0], x_e[idx_0])
    LR1 = LogisticRegression(random_state=0, solver='liblinear')
    LR1.fit(x_c[idx_1], x_e[idx_1])
