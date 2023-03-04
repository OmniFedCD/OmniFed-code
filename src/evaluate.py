# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
import math


"""
 Calculate the f1 score for a point.
 
 @param predict - predict label for the point.
 @param actual - the label of the actual point.
"""
def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


"""
 Adjust predicted labels using given score label and label.
 
 @param score - The ground truth score. A point is labeled as anomaly if its score is lower than the
 @param label - The ground truth label. A point is labeled as anomaly if its score is lower than the
 @param threshold=None - The ground truth label. A point is labeled as anomaly if its score is lower than the threshold.
 @param pred=None - The ground truth label. A point is labeled as anomaly if its score is lower than the threshold.
 @param calc_latency=False - If True calculate latency for anomaly score.
"""
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        # omni算的是重建概率，所以是小于号，这里算的是重建误差，用大于
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


"""
 Calculate a score sequence for a score
 
 @param score - The score to calculate the score for
 @param label - The label of the score
 @param threshold - The threshold of the score.
 @param calc_latency=False - If True then the latency is used to calculate the score
"""
def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


"""
 Search for the best f1 score in the range start end.
 
 @param score - score of the best f1 score.
 @param label - label of the label to search for.
 @param start - The start of the range of the score.
 @param end=None - end of the range of the score.
 @param step_num=1 - step number of the search step.
 @param display_freq=1 - display the best f1 score in the range [ 0 1 ]
 @param verbose=True - If True print out the score and score.
"""
def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] >= m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    # print(m, m_t)
    return m, m_t


"""
 Generate random index between 0 and 1.
 
 @param y_true - A list of true values.
 @param y_pred - preds of the same length as y_true.
"""
def rand_index(y_true, y_pred):
    n = len(y_true)
    a, b = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                a +=1
            elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                b +=1
            else:
                pass
    RI = (a + b) / (n*(n-1)/2)
    return RI

"""
 Get the score of the predicted values of the given labels.
 
 @param y_true - A set of true labels.
 @param y_pred - A list of predicted values.
"""
def get_score(y_true,y_pred):
    ri = rand_index(y_true,y_pred)
    ari = metrics.adjusted_rand_score(y_true,y_pred)#-1~1 1
    ami = metrics.adjusted_mutual_info_score(y_true,y_pred)#-1~1 1
    nmi = metrics.normalized_mutual_info_score(y_true,y_pred)#-1~1 1
    h = metrics.homogeneity_score(y_true,y_pred)
    c = metrics.completeness_score(y_true,y_pred)
    v = metrics.v_measure_score(y_true,y_pred)#0-1 1
    fmi = metrics.fowlkes_mallows_score(y_true,y_pred)#0-1 1
    return ri, ari, nmi, ami, h ,c, v, fmi
    # return nmi
