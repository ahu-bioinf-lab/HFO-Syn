# import numpy as np
# import torch
# from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score
# from scipy.stats import pearsonr
#
# # 分类模型的性能指标的，它的输入参数是真实标签值 yt 和模型预测标签值 yp。
# # 具体来说，它计算了以下指标：ROC曲线下面积（AUC）、精确度-召回率曲线下面积（AUPR）、F1分数、准确率
# def metrics_graph(yp, yt):
#     precision, recall, _, = precision_recall_curve(yt, yp)
#     aupr = -np.trapz(precision, recall)
#     auc = roc_auc_score(yt, yp)
#     # ---f1,acc,recall, specificity, precision
#     #real_score = np.mat(yt)
#     real_score = np.asmatrix(yt)
#     predict_score = np.mat(yp)
#     sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
#     sorted_predict_score_num = len(sorted_predict_score)
#     thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
#     thresholds = np.mat(thresholds)
#     thresholds_num = thresholds.shape[1]
#     predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
#     negative_index = np.where(predict_score_matrix < thresholds.T)
#     positive_index = np.where(predict_score_matrix >= thresholds.T)
#     predict_score_matrix[negative_index] = 0
#     predict_score_matrix[positive_index] = 1
#     TP = predict_score_matrix.dot(real_score.T)
#     FP = predict_score_matrix.sum(axis=1) - TP
#     FN = real_score.sum() - TP
#     TN = len(real_score.T) - TP - FP - FN
#     tpr = TP / (TP + FN)
#     recall_list = tpr
#     precision_list = TP / (TP + FP)
#     f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
#     accuracy_list = (TP + TN) / len(real_score.T)
#     specificity_list = TN / (TN + FP)
#     max_index = np.argmax(f1_score_list)
#     f1_score = f1_score_list[max_index]
#     accuracy = accuracy_list[max_index]
#     specificity = specificity_list[max_index]
#     recall = recall_list[max_index]
#     precision = precision_list[max_index]
#     return auc, aupr, f1_score[0, 0], accuracy[0, 0]  # , recall[0, 0], specificity[0, 0], precision[0, 0]
#
# # 计算回归任务的度量指标
# def regression_metric(ypred, ytrue):
#     rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
#     r2 = r2_score(y_true=ytrue, y_pred=ypred)
#     r, p = pearsonr(ytrue, ypred)
#     return rmse, r2, r

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score
from scipy.stats import pearsonr

def metrics_graph(yp, yt):
    """
    Calculates classification metrics including AUC, AUPR, F1-score, and accuracy.

    Args:
        yp (np.ndarray): Predicted scores.
        yt (np.ndarray): True labels.

    Returns:
        tuple: (AUC, AUPR, F1-score, Accuracy)
    """
    # Calculate AUC and AUPR using scikit-learn
    precision, recall, _ = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)

    # --- f1, acc, recall, specificity, precision calculation

    # Use np.array and reshape to ensure correct dimensions
    real_score = np.array(yt).reshape(1, -1)
    predict_score = np.array(yp).reshape(1, -1)

    # Calculate metrics at various thresholds
    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = thresholds.reshape(1, -1)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))

    # Compare predictions with thresholds
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    # Use standard array operations and the @ operator for matrix multiplication
    TP = predict_score_matrix @ real_score.T
    FP = predict_score_matrix.sum(axis=1, keepdims=True) - TP
    FN = real_score.sum() - TP
    TN = real_score.shape[1] - TP - FP - FN

    tpr = TP / (TP + FN + 1e-10)
    recall_list = tpr
    precision_list = TP / (TP + FP + 1e-10)
    f1_score_list = 2 * TP / (real_score.shape[1] + TP - TN + 1e-10)
    accuracy_list = (TP + TN) / real_score.shape[1]
    specificity_list = TN / (TN + FP + 1e-10)

    # Find metrics at the best F1-score threshold
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    # The result is a 1x1 array, extract the scalar value
    return auc, aupr, f1_score.item(), accuracy.item()


# 计算回归任务的度量指标
def regression_metric(ypred, ytrue):
    """
    Calculates regression metrics including RMSE, R2, and Pearson correlation coefficient.
    Args:
        ypred (np.ndarray): Predicted values.
        ytrue (np.ndarray): True values.

    Returns:
        tuple: (RMSE, R2, Pearson correlation coefficient)
    """
    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    r, _ = pearsonr(ytrue, ypred)
    return rmse, r2, r