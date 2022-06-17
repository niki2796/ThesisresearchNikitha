import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


LABELS = ["Normal","Anomaly"]

bag =500
num_runs = 10 #no. of ensembles
scores = []
rocs = []

for i in range(num_runs):

    print('Run: ',i)

    a = np.load('cardio.npz')
    x = a['x']
    np.random.shuffle(x)
    x = x[:bag]
    y = np.zeros(len(x))
    tx = a['tx']
    ty = a['ty']
    nb_epoch = 100
    batch_size = 64
    input_dim = x.shape[1]
    encoding_dim = 14
    hidden_dim_1 = int(encoding_dim / 2) #
    hidden_dim_2=4
    learning_rate = 1e-7

    def autoencoder_model():
        #input Layer
        input_layer = tf.keras.layers.Input(shape=(input_dim, ))
        #Either normalise or remove tanh from last layer
        #check elu activation function
        #Encoder
        encoder = tf.keras.layers.Dense(encoding_dim, activation="elu",
                                        activity_regularizer=tf.keras.regularizers.l2(learning_rate))(input_layer)
        encoder=tf.keras.layers.Dropout(0.2)(encoder)
        encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
        encoder = tf.keras.layers.Dense(hidden_dim_2, activation=tf.nn.leaky_relu)(encoder)

        # Decoder
        decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
        decoder=tf.keras.layers.Dropout(0.2)(decoder)
        decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)
        decoder = tf.keras.layers.Dense(input_dim, activation='elu')(decoder)

        #Autoencoder
        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
        autoencoder.summary()
        return autoencoder

    autoencoder = autoencoder_model()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')

    history = autoencoder.fit(x, x,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        #validation_data=(tx, tx),
                        #filter anomaly from tx and then validate, #use x instead of tx to validate
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[early_stop]
                        ).history

    '''
    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.plot(history['val_loss'], linewidth=2, label='Test')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.ylim(ymin=0.70,ymax=1)
    plt.savefig('lossplots.png')
    '''

    #reconstruct
    test_x_predictions = autoencoder.predict(tx)
    train_x_predictions = autoencoder.predict(x)
    mse = np.mean(np.power(tx - test_x_predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': ty})
    threshold_fixed = 0.65
    '''
    #plot reconstruction
    groups = error_df.groupby('True_class')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                label= "Anomaly" if name == 1 else "Normal")
    ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for normal and fraud data")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    '''

    #evaluation
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    error_df['pred'] =pred_y
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    #plt.figure(figsize=(4, 4))
    #sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    #plt.title("Confusion matrix")
    #plt.ylabel('True class')
    #plt.xlabel('Predicted class')
    #roc = roc_auc_score(ty, pred_y, average=None)
    roc = roc_auc_score(ty, mse, average=None)

    scores.append(mse)
    rocs.append(roc)

    '''    
    #plot roc
    fig=plt.figure()
    fpr, tpr, _ = roc_curve(ty, mse)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='area = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='lower right')
    #plt.show()
    plt.close('all')
    '''
scores = np.array(scores)
score = np.mean(scores, axis = 0)

threshold_fixed = 0.65
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
error_df['pred'] =pred_y
conf_matrix = confusion_matrix(error_df.True_class, pred_y)
plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig('cm.png')
roc = roc_auc_score(ty, score, average=None)
print(roc)

#plot roc
fig=plt.figure()
fpr, tpr, _ = roc_curve(ty, score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='area = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='lower right')
plt.savefig('roc.png')
plt.show()




