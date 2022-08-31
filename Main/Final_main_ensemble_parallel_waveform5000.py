import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend as K
from sklearn import preprocessing
from threshold_calc import *
from scipy import stats
from tqdm import tqdm
import tensorflow_probability as tfp

LABELS = ["Normal","Anomaly"]

tf.random.set_seed(1121)
np.random.seed(1121)

def autoencoder_model(num_parallel, input_dim, encoding_dim, hidden_dim_1, hidden_dim_2):
    input_layer = tf.keras.layers.Input(shape=(input_dim, ))
    encoder = [None]*num_parallel
    decoder = [None]*num_parallel

    for i in range(num_parallel):

        encoder[i] = tf.keras.layers.Dense(encoding_dim, activation="elu",
                                        activity_regularizer=tf.keras.regularizers.l2(learning_rate))(input_layer)
        encoder[i]=tf.keras.layers.Dropout(0.2)(encoder[i])
        encoder[i] = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder[i])
        encoder[i] = tf.keras.layers.Dense(hidden_dim_2, activation=tf.nn.leaky_relu)(encoder[i])

        # Decoder
        decoder[i] = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder[i])
        decoder[i]=tf.keras.layers.Dropout(0.2)(decoder[i])
        decoder[i] = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder[i])
        decoder[i] = tf.keras.layers.Dense(input_dim, activation='elu')(decoder[i])

    #Autoencoder
    decoder = [K.expand_dims(a,axis=2) for a in decoder]
    decoder = tf.keras.layers.concatenate(decoder,axis=2)
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()
    return autoencoder

def autoenc_train(training_loss, x, nb_epoch, batch_size,num_parallel, input_dim, encoding_dim, hidden_dim_1, hidden_dim_2):

    autoencoder = autoencoder_model(num_parallel, input_dim, encoding_dim, hidden_dim_1, hidden_dim_2)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    autoencoder.compile(loss=training_loss,
                        optimizer='adam')

    history = autoencoder.fit(x, x,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[early_stop]
                        ).history
    return autoencoder

def autoenc_predict(autoencoder,tx, ty, predict_loss):
    test_x_predictions = autoencoder.predict(tx)
    parallel_rocs = []
    for ind in range(test_x_predictions.shape[-1]):
        score_each = predict_loss(tx, test_x_predictions[:,:,ind], return_mean=False, parallel_loss= False)
        parallel_rocs.append(roc_auc_score(ty, score_each, average=None))
    parallel_rocs = np.array(parallel_rocs)
    score = predict_loss(tx, test_x_predictions, return_mean=False)
    error_df = pd.DataFrame({'Reconstruction_error': score, 'True_class': ty})
    #evaluation
    threshold_fixed = get_optimal_threshold(ty, score, steps=100, return_metrics=False, flag = "f1_score")
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    error_df['pred'] =pred_y
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig('cm.png')
    plt.close('all')
    roc = roc_auc_score(ty, score, average=None)
    return roc, score, error_df, threshold_fixed, parallel_rocs

def my_mse(tx, test_x_predictions, return_mean = False, parallel_loss = True):
    if parallel_loss == False:
        return np.mean((test_x_predictions - tx) ** 2, axis=-1)
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
    return K.mean(K.mean(q, axis = -1))
    #return K.mean(q)

def max_loss_1(a,b, return_mean=True):
    q=b
    pd=[i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0,len(pd))
    q=K.permute_dimensions(q,tuple(pd))
    q=(q-a)**2
    if return_mean == False:
        return K.max(K.mean(q, axis = -1), axis = 0)
    #return K.mean(K.max(K.mean(q, axis = -1), axis = 0))
    return K.max(K.mean(K.mean(q, axis = -1), axis = 1))

def min_loss_1(a,b, return_mean=True):
    q=b
    pd=[i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0,len(pd))
    q=K.permute_dimensions(q,tuple(pd))
    q=(q-a)**2
    if return_mean == False:
        return K.min(K.mean(q, axis = -1), axis = 0)
    return K.mean(K.min(K.mean(q, axis = -1), axis = 0))
    #return K.min(K.mean(K.mean(q, axis = -1), axis = 1))

def median_loss_1(a,b, return_mean=True):
    q=b
    pd=[i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0,len(pd))
    q=K.permute_dimensions(q,tuple(pd))
    q=(q-a)**2
    if return_mean == False:
        return tfp.stats.percentile(K.mean(q, axis = -1), q=50, axis = 0)
    return tfp.stats.percentile(K.mean(K.mean(q, axis = -1), axis = 1), q=50)
    # return K.mean(tfp.stats.percentile(K.mean(q, axis = -1), q=50, axis = 0))

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

def median_loss_2(a,b, return_mean=True):
    q=b
    pd=[i for i in range(len(q.shape))]
    pd.remove(pd[-1])
    pd.insert(0,len(pd))
    q=K.permute_dimensions(q,tuple(pd))
    q=K.square(a - tfp.stats.percentile(q, q=50, axis = 0))
    if return_mean == False:
        return K.mean(q, axis=-1)
    return K.mean(q)

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


if __name__ == '__main__':
    num_runs = 5 # no. of roc runs
    start = 3
    end = 63
    skip = 3
    nb_epoch = 200
    batch_size = 64
    data_set = 'waveform-5000.npz'
    a = np.load(data_set)
    x = a['x'].astype(np.float32)
    bx = x.shape
    bx = bx[-1]
    tx = a['tx'].astype(np.float32)
    ty = a['ty']
    x = preprocessing.normalize(x, norm='l2')
    bag = len(x)
    input_dim = x.shape[1]
    encoding_dim = bx // 2
    hidden_dim_1 = int(encoding_dim / 2)
    hidden_dim_2 = hidden_dim_1 - 2
    learning_rate = 1e-7
    best_roc = 0

    text_to_file = " Total number of records: " + str(len(x)+len(tx)) + \
                   "\n Shape of training data: " + str([str(val) for val in x.shape])  + \
                   "\n Shape of test data: " + str([str(val) for val in tx.shape]) + \
                   "\n Number of anomalies in text: " + str(ty.sum())

    text_file = open("file_details.txt", "w")
    text_file.write(text_to_file)
    text_file.close()

    #tr_loss = [median_loss_1, median_loss_1]
    #pr_loss = [my_mse, median_loss_1]
    tr_loss = [loss_1,  max_loss_1, min_loss_1, median_loss_1, loss_2, median_loss_2, loss_3, loss_4]
    pr_loss = [my_mse, my_mse, my_mse, my_mse,  my_mse, my_mse, my_mse, my_mse]
    store_values = np.zeros([int((end-start)/skip),len(tr_loss)])
    store_sd = np.zeros([int((end-start)/skip),len(tr_loss)])
    store_sd_par = np.zeros([int((end - start) / skip), len(tr_loss), num_runs])
    store_average_corr = np.zeros([int((end-start)/skip),len(tr_loss)])

    count = 0
    for itr in tqdm(range(start, end, skip)):
        num_parallel = itr
        print("No. of parallel auto encoders =", itr)

        for itr_loss in tqdm(range(0, len(tr_loss)),leave=False):
            training_loss = tr_loss[itr_loss]
            predict_loss = pr_loss[itr_loss]

            scores = []
            rocs = []
            std_roc_par = []
            roc_matrix = []
            for i in range(num_runs):
                print('Run: ',i)
                x = a['x'].astype(np.float32)
                x = preprocessing.normalize(x, norm='l2')
                np.random.shuffle(x)
                x = x[:bag]
                y = np.zeros(len(x))
                tx = a['tx'].astype(np.float32)
                tx = preprocessing.normalize(tx, norm='l2')
                ty = a['ty']
                autoencoder = autoenc_train(training_loss, x, nb_epoch, batch_size,num_parallel, input_dim, encoding_dim, hidden_dim_1, hidden_dim_2)
                #reconstruct
                roc, score, error_df, threshold_fixed, parallel_rocs = autoenc_predict(autoencoder,tx, ty, predict_loss)
                std_roc_par.append(np.std(parallel_rocs))
                roc_matrix.append(parallel_rocs)
                scores.append(score)
                rocs.append(roc)
            scores = np.array(scores)
            score = np.mean(scores, axis = 0)
            #pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
            #error_df['pred'] =pred_y
            #conf_matrix = confusion_matrix(error_df.True_class, pred_y)
            #roc = roc_auc_score(ty, score, average=None)
            roc_matrix = np.array(roc_matrix)
            corr_matrix = np.corrcoef(roc_matrix.T)

            #pd.DataFrame(corr_matrix).to_csv('corr_mat_{}_{}.csv'.format(itr,str(tr_loss[i].__name__)))
            std_roc_par_array = np.array(std_roc_par)
            roc = np.mean(np.array(rocs))
            print(roc)
            roc_sd = np.std(np.array(rocs))
            store_values[count][itr_loss] = roc
            store_sd[count][itr_loss] = roc_sd
            store_sd_par[count][itr_loss] = std_roc_par_array
            store_average_corr[count][itr_loss] = ((np.sum(corr_matrix) - np.trace(corr_matrix))/((itr*itr)-itr))
            if itr == (end - skip):
                if roc > best_roc:
                    best_roc = roc
                    best_corr_matrix = corr_matrix
                    best_train_loss_60 = str(tr_loss[itr_loss].__name__)
        count += 1
        np.save('stored_val.npy', store_values)
        np.save('stored_sd.npy', store_sd)
        np.save('stored_sd_par.npy', store_sd_par)

        x_ax = np.array([i for i in range(start, end, skip)])

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

        fig = plt.figure()
        for i in range(0, store_values.shape[1]):
            plt.plot(x_ax, store_values[:, i], label=str(tr_loss[i].__name__) + '-' + str(pr_loss[i].__name__))
        plt.title('ROCs of different losses vs number of ensembles')
        plt.xlabel('Number of Ensembles')
        plt.ylabel('ROC')
        plt.legend()
        plt.show()
        plt.savefig('loss_ens.png')

        fig1 = plt.figure()
        for i in range(0, store_sd.shape[1]):
            plt.plot(x_ax, store_sd[:, i], label=str(tr_loss[i].__name__) + '-' + str(pr_loss[i].__name__))
        plt.title('Standard deviation of ROCs')
        plt.xlabel('Number of Ensembles')
        plt.ylabel('ROC standard deviation')
        plt.legend()
        plt.show()
        plt.savefig('ROC_SD.png')

        fig2 = plt.figure()
        for i in range(0, 4):
            plt.plot(x_ax, store_values[:, i], label=str(tr_loss[i].__name__) + '-' + str(pr_loss[i].__name__))
        plt.title('ROC comparison for all the variants of loss1')
        plt.xlabel('Number of Ensembles')
        plt.ylabel('ROC')
        plt.legend()
        plt.show()
        plt.savefig('loss1_comparison.png')

        fig3 = plt.figure()
        for i in range(4, 6):
            plt.plot(x_ax, store_values[:, i], label=str(tr_loss[i].__name__) + '-' + str(pr_loss[i].__name__))
        plt.title('ROC comparison for all the variants of loss2')
        plt.xlabel('Number of Ensembles')
        plt.ylabel('ROC')
        plt.legend()
        plt.show()
        plt.savefig('loss2_comparison.png')

        fig4 = plt.figure()
        for i in range(0, store_average_corr.shape[1]):
            plt.plot(x_ax, store_average_corr[:, i], label=str(tr_loss[i].__name__) + '-' + str(pr_loss[i].__name__))
        plt.title('Average correlation between the ROCs')
        plt.xlabel('Number of Ensembles')
        plt.ylabel('Average correlation of ROC')
        plt.legend()
        plt.show()
        plt.savefig('Average_ROC.png')

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
    p_sig.to_csv('significance_test_result.csv')

    fig5 = plt.figure()
    ax = sns.heatmap(best_corr_matrix, linewidth=0.5, cmap ="Blues")
    ax.set_title('Heatmap of ' + best_train_loss_60 + ' correlation matrix')
    plt.savefig('Heatmap_corrmat_'+ best_train_loss_60 + '.png')



