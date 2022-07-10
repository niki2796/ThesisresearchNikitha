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
store_values = np.load('stored_val.npy')
tr_loss = [loss_1, loss_1,  max_loss_1,  max_loss_1, min_loss_1, min_loss_1, median_loss_1, median_loss_1, loss_2, median_loss_2, loss_3, loss_4]
pr_loss = [loss_1, my_mse,  max_loss_1, my_mse, min_loss_1, my_mse, median_loss_1, my_mse,  my_mse, my_mse, my_mse, my_mse]

a_dict = {}

for i in range(0, store_values.shape[1]):
    for j in range(i+1, store_values.shape[1]):
        a_dict[str(tr_loss[i].__name__)
               + '.'
               + str(pr_loss[i].__name__)
               + '-'
               + str(tr_loss[j].__name__) +
               '.'
               + str(pr_loss[j].__name__)] = [stats.ttest_ind(store_values[:, i],store_values[:, j]).pvalue]

p_sig = pd.DataFrame.from_dict(a_dict, orient='index')
not_sig = p_sig[p_sig[0]>0.05]
p_sig.to_csv('significance_test_result.csv')
not_sig.to_csv('Non_significant_losses.csv')
