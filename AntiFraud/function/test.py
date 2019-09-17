import numpy as np
from function.data_set import load_dataset_to_csv
from function.lr import sigmoid
from function.k_means import dist_euclidean


def load_w(n):
    w = np.mat(np.ones((n, 1)))
    i = 0
    with open('../txt/w.txt', 'r') as f:
        for line in f:
            w[i] = float(line)
            i += 1
    return w
# 获取w系数矩阵,n为列维度


def load_center(n, k):
    center = np.mat(np.ones((k, n)))
    i = 0
    j = 0
    with open('../txt/center.txt', 'r') as f:
        for line in f:
            center[i, j] = float(line)
            j += 1
            if j == n:
                i += 1
                j = 0
    return center
# 获取质心矩阵


def load_prob(k):
    prob = np.mat(np.ones((k, 1)))
    i = 0
    with open('../txt/prob.txt', 'r') as f:
        for line in f:
            prob[i] = float(line)
            i += 1
    return prob
# 获取聚类负类概率


def class_if_lr(data_mat, w):
    prob = sigmoid(data_mat * w)
    for i in range(len(prob)):
        if prob[i] > 0.5:
            prob[i] = 1.0
        else:
            prob[i] = 0.0
    return prob
# lr分类，返回某一行类别


def class_prob_k(data_mat, center_mat, prob, k):
    m = data_mat.shape[0]
    prob_mat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        best_dist = np.inf
        for j in range(k):
            dist = dist_euclidean(center_mat[j], data_mat[i, :])
            if dist < best_dist:
                best_dist = dist
                prob_mat[i] = prob[j]
    return prob_mat
# 根据k-means的聚类质心，获得每个数据的所属类别，并判断为正类的概率


def test(k):
    data, labels = load_dataset_to_csv('testdata')
    m, n = np.shape(data)
    data_mat = np.mat(data)
    labels_mat = np.mat(labels).transpose()

    w = load_w(n)
    result_lr = sigmoid(data_mat * w)
    center = load_center(n, k)
    prob_k = load_prob(k)
    result_k = class_prob_k(data_mat, center, prob_k, k)

    result_end = np.multiply(result_lr, result_k)
    error = np.mat(np.zeros((2, 2)))
    for i in range(m):
        if labels_mat[i] == 0 and result_end[i] < 0.25:
            error[0, 0] += 1
        if labels_mat[i] == 0 and result_end[i] > 0.25:
            error[1, 0] += 1
        if labels_mat[i] == 1 and result_end[i] > 0.25:
            error[1, 1] += 1
        if labels_mat[i] == 1 and result_end[i] < 0.25:
            error[0, 1] += 1
    print('--混淆矩阵已生成--')
    print(error)
    # 制作简易的混淆矩阵
    return error, labels_mat, result_end
# 测试返回结果矩阵、分类结果、判断概率


def rate(error):
    rate_dict = {}
    tp = error[0, 0]  # 正确判为正常的数目，true positive
    tn = error[1, 1]  # 正确判为欺诈的数目，true negative
    fn = error[1, 0]  # 正常数据判为欺诈的数目，false negative
    fp = error[0, 1]  # 欺诈数据判为正常的数目，false positive
    tpr = tp/(tp + fn)  # tp/(tp + fn)，真正类率
    fpr = fp/(fp + tn)  # fp/(fp + tn)，负正类率
    accuracy = (tn + tp)/(tp + tn + fn + fp)  # 准确率
    precision = tp/(tp + fp)  # 精确率，判断正常的数据中，正确的比率
    recall = tp/(tp + fn)  # 召回率，判断正常且正确的数据，占所有正常数据的比率
    f1 = precision * recall * 2/(precision + recall)  # f1值，精确率与召回率的综合
    rate_dict['Accuracy准确率'] = accuracy
    rate_dict['Precision精确率'] = precision
    rate_dict['Recall召回率'] = recall
    rate_dict['F1'] = f1
    print('--评价完毕--')
    print(rate_dict)
    return rate_dict
# 评价结果，以dict形式


if __name__ == '__main__':
    error, lb, re = test(4)
    # print(lb, re)
    rate(error)


'''   由两次分类直接判断结果、粗略分类
error_k = np.zeros((k, 2))
    for i in range(m):
        for j in range(k):
            if result_k[i, 0] == j and labels_mat[i] == 0:
                error_k[j, 0] += 1
            if result_k[i, 0] == j and labels_mat[i] == 1:
                error_k[j, 1] += 1
    # 获取聚类类结果矩阵error_k，4x2表示4类和2个实际类别

    result_end = np.mat(np.zeros((m, 1)))
    error = np.mat(np.zeros((2, 2)))
    for j in range(k):
        if error_k[j, 0] == max(error_k[:, 0]):
            most_class = j
    # 获取正常数据最多的类别
    for i in range(m):
        if result_k[i] == most_class:
            result_end[i] = 0
        else:
            if result_lr[i] == 0:
                result_end[i] = 0
            if result_lr[i] == 1:
                result_end[i] = 1
    # 先用聚类结果筛选，再用回归确定结果
'''









