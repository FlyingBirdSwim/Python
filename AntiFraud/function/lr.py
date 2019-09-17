import numpy as np
from function.data_set import load_dataset_to_csv


def sigmoid(x):
    r = 1.0/(1+np.exp(-x))
    return r
# 判断函数


def cost_func(data_mat, labels_mat, w):
    sum = 0.0
    m = data_mat.shape[0]
    for i in range(m):
        lb_pre = sigmoid(data_mat[i] * w)
        if labels_mat[i] == 0:
            sum += np.log(1-lb_pre)
        if labels_mat[i] == 1:
            sum += np.log(lb_pre)
    # 计算每一列的预测类别与实际类别的差值，然后求和
    return -1/m * sum
# 代价函数


def logistic_regression(data_list, labels_list):
    data_mat = np.mat(data_list)
    labels_mat = np.mat(labels_list).transpose()
    # 转化为mat矩阵
    m, n = np.shape(data_mat)
    # 获取矩阵维度，m为行，n为列
    w = np.mat(np.full((n, 1), 0.5))
    # w作为函数的系数矩阵，ones()创建一个n行1列的矩阵
    r = 0.001
    # r步长
    accuracy = 0.000001

    count = 0
    while True:
        old_cost = cost_func(data_mat, labels_mat, w)
        h = sigmoid(data_mat * w)
        # 预测类别h
        error = h - labels_mat
        # error表示实际值和预测值的差
        w = w - r * data_mat.transpose() * error
        # 梯度计算
        new_cost = cost_func(data_mat, labels_mat, w)
        if new_cost < 0.1 or np.fabs(new_cost - old_cost) < accuracy:
            print("代价函数迭代到最小值，退出！")
            print("收敛到:", new_cost)
            break
        print("迭代第", count, "次!")
        print("本次代价为：", new_cost)
        print("代价函数上一次的差:", (new_cost - old_cost))
        count += 1
    return w
# 改进的梯度下降算法


def w_save_txt(w, filename):
    w = np.array(w)
    st = ''
    for i in range(len(w)):
        for j in range(len(w[0])):
            st = st + str(w[i][j])
            st = st + '\n'
    file = open(filename, 'w')
    file.write(st)
    file.close()
# 保存w系数文本


def run_lr():
    data, labels = load_dataset_to_csv('traindata')
    a = logistic_regression(data, labels)
    w_save_txt(a, '../txt/w.txt')
    print('参数保存成功')


if __name__ == '__main__':
    run_lr()

