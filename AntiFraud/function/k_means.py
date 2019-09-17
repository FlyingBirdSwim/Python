import numpy as np
from function.data_set import load_dataset_to_csv


def dist_euclidean(vec_a, vec_b):
    r = np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))
    return r
# 欧氏距离函数，参数对应矩阵的每一行


def rand_center(data, k):
    n = np.shape(data)[1]
    centers = np.mat(np.zeros((k, n)))
    # 初始化质心矩阵
    for j in range(n):
        j_min = min(data[:, j])
        j_max = max(data[:, j])
        j_range = float(j_max - j_min)
        # 生成随机数 ，k行（代表k个中心点）
        centers[:, j] = j_min + j_range * np.random.rand(k, 1)
        # random.rand(k,1)->k行一列的数据 randej标量
    return centers
# 获取初始质心，参数data为mat矩阵


def k_means(data, k):
    data = np.mat(data)
    m = np.shape(data)[0]
    cluster_mat = np.mat(np.zeros((m, 2)))
    centers = rand_center(data, k)
    # 生成类别纪录矩阵和初始质心矩阵

    changed = True
    while changed:
        changed = False
        # 循环每个点，计算他与每个质心的位置
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            # 初始化最小距离和对应索引
            for j in range(k):
                j_dist = dist_euclidean(centers[j, :], data[i, :])
                if j_dist < min_dist:
                    min_dist = j_dist
                    min_index = j
                # 找第i个数据点所属的簇，且计算距离
            if cluster_mat[i, 0] != min_index:
                changed = True
                # 更新这个点到质心的索引及误差
            cluster_mat[i, :] = min_index, min_dist**2
            # 将i行所属的簇、与质心的距离记录到矩阵

        for cent in range(k):
            # 查找这个 cent 簇所有的点
            points_in_cluster = data[np.nonzero(cluster_mat[:, 0] == cent)[0]]
            # np.nonzero返回非0的一个2维tuple数组，第一列为数组，第二列为数据类型
            # 第cent个簇所有点
            if len(points_in_cluster) != 0:
                centers[cent, :] = np.mean(points_in_cluster, axis=0)
            # 对于这个簇中每个点的列取均值，更新中心点centers[cent]
    return centers, cluster_mat
    # k均值聚类
# k均值聚类


def bi_k_means(data, k):
    data = np.mat(data)
    m = np.shape(data)[0]
    cluster_mat = np.mat(np.zeros((m, 2)))
    # 纪录所属簇和距离的矩阵
    center0 = np.mat(np.mean(data, axis=0))
    # 初始质心点，每行均值
    center_list = [center0]
    # 初始质心list只有一个质心
    for j in range(m):
        cluster_mat[j, 1] = dist_euclidean(data[j], center0)

    while (len(center_list)<k):
        lowest_see = np.inf
        # 初始最小误差为正无穷
        for i in range(len(center_list)):
            cluster_i_data = data[np.nonzero(cluster_mat[:, 0] == i)[0], :]
            # 取出属于第i类的数据集合
            center_i_mat, cluster_i_mat = k_means(cluster_i_data, 2)
            # 第i类的质心和记录矩阵
            sse_split = sum(cluster_i_mat[:, 1])
            # 划分后两个类的误差平方和
            sse_not_split = sum(cluster_mat[np.nonzero(cluster_mat[:, 0] != i)[0], 1])
            # 记录矩阵中不属于第i类的误差平方和
            total_split = sse_split + sse_not_split
            # 总误差
            if total_split < lowest_see:
                best_cent_to_split = i
                best_new_cent = center_i_mat
                best_clust_ass = cluster_i_mat
                lowest_see = total_split
                # 以此划分第i类，将i类划分为两个质心，即二分
                # 复制记录矩阵，更新误差

        # 2分k聚类返回类别0或1，需要把1换成当前簇数目，以免造成重复
        best_clust_ass[np.nonzero(best_clust_ass[:, 0] == 1)[0], 0] = len(center_list)
        best_clust_ass[np.nonzero(best_clust_ass[:, 0] == 0)[0], 0] = best_cent_to_split
        # 重新划分记录矩阵的类别，0->i,1->len(center_list)
        # 0->i类在之后会继续进行二分划分，直到满足循环条件
        print('===:', best_cent_to_split)
        center_list[best_cent_to_split] = best_new_cent[0, :]
        # 在list中更新第i类的质心为前一个质心
        center_list.append(best_new_cent[1, :])
        # 将划分的后一个质点添加入结果矩阵
        cluster_mat[np.nonzero(cluster_mat[:, 0] == best_cent_to_split)[0], :] = best_clust_ass
    return center_list, cluster_mat
# 二分k均值聚类


def get_k_prob(labels_list, cluster_mat, k):
    labels_mat = np.mat(labels_list).transpose()
    m = labels_mat.shape[0]
    error_k = np.zeros((k, 2))
    for i in range(m):
        for j in range(k):
            if cluster_mat[i, 0] == j and labels_mat[i] == 0:
                error_k[j, 0] += 1
            if cluster_mat[i, 0] == j and labels_mat[i] == 1:
                error_k[j, 1] += 1
    # 获得混淆矩阵
    prob_arr = np.zeros(k)
    for i in range(k):
        prob_arr[i] = error_k[i, 1]/(error_k[i, 0] + error_k[i, 1])
    # print(error_k, prob_arr)
    return prob_arr
# 输出每一类为负类的概率


def center_save_txt(center_list, filename):
    st = ''
    for i in range(len(center_list)):
        array = np.array(center_list[i])
        for m in range(len(array)):
            for n in range(len(array[0])):
                st = st + str(array[m][n])
                st = st + '\n'
    file = open(filename, 'w')
    file.write(st)
    file.close()
# 保存分类簇质心点文本


def cluster_save_txt(cluster_mat, filename):
    st = ''
    for i in range(len(cluster_mat)):
        st = st + str(cluster_mat[i, 0])
        st = st + '\n'
    file = open(filename, 'w')
    file.write(st)
    file.close()
# 保存数据集分类情况


def prob_save_txt(prob_arr, filename):
    st = ''
    for i in range(len(prob_arr)):
        st = st + str(prob_arr[i])
        st = st + '\n'
    file = open(filename, 'w')
    file.write(st)
    file.close()
# 保存分类概率情况


def run_k_means(k):
    data, labels = load_dataset_to_csv('traindata')
    center, cluster = bi_k_means(data, k)
    prob = get_k_prob(labels, cluster, k)
    center_save_txt(center, '../txt/center.txt')
    prob_save_txt(prob, '../txt/prob.txt')
# 输入k值，进行聚类


if __name__ == '__main__':
    run_k_means(4)
