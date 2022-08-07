import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend as K

def my_mse(tx, test_x_predictions, return_mean = False):
    return np.mean((np.mean(test_x_predictions, axis=-1) - tx) ** 2, axis=-1)

def loss_1(a,b, return_mean=True):
    q=b
    pd=[i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0,len(pd))
    q=K.permute_dimensions(q,tuple(pd))
    q=(q-a)**2
    if return_mean == False:
        return K.mean(q, axis=(0, -1))
    return K.mean(q)

def s_loss_1(a, b):
    q = b
    pd = [i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0, len(pd))
    q = K.permute_dimensions(q, tuple(pd))
    q = (q - a) ** 2
    return K.mean(q)

def loss_2(a,b, return_mean=True):
    q=b
    pd=[i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0,len(pd))
    q=K.permute_dimensions(q,tuple(pd))
    q=K.square(a -  K.mean(q, axis=0))
    if return_mean == False:
        return K.mean(q, axis=-1)
    return K.mean(q)

def s_loss_2(a, b):
    return K.mean(K.square(K.mean(b, axis=2) - a))


def loss_3(a,b):
    q=b
    pd=[i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0, len(pd))
    q=K.permute_dimensions(q,tuple(pd))

    outdim = q.shape[0]

    q = (q - a)

    adl = None

    for i in range(outdim):
        for j in range(i + 1, outdim):
            ac = K.abs(K.mean(q[i] * q[j]))
            if adl is None:
                adl = ac
            else:
                adl += ac

    div = outdim * (outdim - 1) / 2
    return K.mean(adl) / div

def loss_4(a,b, return_mean=True):
    q=b
    pd=[i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0, len(pd))
    q=K.permute_dimensions(q,tuple(pd))

    outdim = q.shape[0]

    q = (q - a) ** 2  # rms
    q = K.mean(q, axis=-1)  # mean

    adl = None

    for i in range(outdim):
        for j in range(i + 1, outdim):
            ac = K.abs(K.mean(q[i] * q[j]))
            if adl is None:
                adl = ac
            else:
                adl += ac

    return adl

LABELS = ["Normal","Anomaly"]

tf.random.set_seed(1121)
np.random.seed(1121)

store_values = np.load('stored_val.npy')
start = 3
end = 24
skip = 3
tr_loss = [loss_1, loss_1, loss_2, loss_2, loss_3, loss_4]
pr_loss = [my_mse, loss_1, my_mse, loss_2, my_mse, my_mse]
x_ax = np.array([i for i in range(start, end, skip)])
fig = plt.figure()
for i in range(0, store_values.shape[1]):
    plt.plot(x_ax, store_values[:, i], label=str(tr_loss[i].__name__) + '-' + str(pr_loss[i].__name__))

plt.title('ROCs of different losses vs number of ensembles')
plt.xlabel('Number of Ensembles')
plt.ylabel('ROC')
plt.legend()
plt.show()
plt.savefig('loss_ens.png')