import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from threshold_calc import *
from Final_main_ensemble_parallel import *
store_sd_par = np.load('stored_sd_par.npy')
tr_loss = [loss_1,  max_loss_1, min_loss_1, median_loss_1, loss_2, median_loss_2, loss_3, loss_4]
pr_loss = [my_mse, my_mse, my_mse, my_mse,  my_mse, my_mse, my_mse, my_mse]
x_ax = np.array([i for i in range(3, 63, 3)])
fig, axs = plt.subplots(4,2)
i_ind = 0
j_ind = 0
for i in range(store_sd_par.shape[1]):
    for j in range(store_sd_par.shape[-1]):
        axs[i_ind,j_ind].plot(x_ax, store_sd_par[:,i,j])
    axs[i_ind, j_ind].set_title(str(tr_loss[i].__name__) )
    axs[i_ind, j_ind].set(xlabel = 'Number of Ensembles')
    i_ind += 1
    if i_ind == 4:
        j_ind += 1
        i_ind = 0
fig.suptitle('Number of Ensembles vs Parallel ROC Std')
fig.tight_layout()
fig.savefig('std_par_roc.png')
