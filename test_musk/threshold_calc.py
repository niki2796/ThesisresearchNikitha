import numpy as np
import sys
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import roc_auc_score

def get_accuracy_precision_recall_fscore(y_true, y_pred):
        # warn_for=() avoids log warnings for any result being zero
        precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
        roc = roc_auc_score(y_true, y_pred, average=None)
        if precision == 0 and recall == 0:
            f01_score = 0
        else:
            f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.05)
        return precision, recall, f_score, f01_score, roc

def get_optimal_threshold(y_test, score, steps=100, return_metrics=False, flag = "f1_score"):
    maximum = np.nanmax(score)
    minimum = np.nanmin(score)
    threshold = np.linspace(minimum, maximum, steps)
    metrics = list(get_metrics_by_thresholds(y_test, score, threshold))
    metrics = np.array(metrics).T
    anomalies, prec, rec, f_score, f01_score, roc =  metrics
    if return_metrics:
        return anomalies, prec, rec, f_score, f01_score, threshold, roc
    else:
        if flag == "f1_score":
            return threshold[np.argmax(f_score)]
        else:
            return threshold[np.argmax(roc)]

        #return np.nanmean(score) + (2*np.nanstd(score))


def get_metrics_by_thresholds(y_test, score, thresholds):
    for threshold in thresholds:
        anomaly = binarize(score, threshold=threshold)
        metrics = get_accuracy_precision_recall_fscore(y_test, anomaly)
        yield (anomaly.sum(), *metrics)

def binarize(score, threshold=None):
    threshold = threshold if threshold is not None else threshold(score)
    score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
    return np.where(score >= threshold, 1, 0)
