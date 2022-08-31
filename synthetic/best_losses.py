import numpy as np
import pandas as pd
from Final_main_ensemble_parallel_elevators import *

stored_roc_cardio = np.load('stored_val.npy')
data_name = 'synt'

tr_loss = [loss_1, max_loss_1, min_loss_1, median_loss_1, loss_2, median_loss_2, loss_3, loss_4]
pr_loss = [my_mse, my_mse, my_mse, my_mse, my_mse, my_mse, my_mse, my_mse]

loss_diff  = stored_roc_cardio - np.expand_dims(stored_roc_cardio[:,0], axis=1)
loss_diff = np.mean(loss_diff, axis=0)

loss_diff_position = np.where(loss_diff>0)

df_dict={}
for i in  loss_diff_position[0]:
    df_dict[tr_loss[i].__name__] = [loss_diff[i], data_name]


df = pd.DataFrame.from_dict(df_dict, orient='index', columns =['Average ROC Distance', 'Data Name']).sort_values(by='Average ROC Distance', ascending=False)
df.to_csv('Best_roc_loss_synt.csv')
