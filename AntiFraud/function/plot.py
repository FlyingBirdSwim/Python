import pandas as pd
import matplotlib.pyplot as plt
from function.test import test
from sklearn.metrics import roc_curve, auc


def class_info():
    data = pd.read_csv('../csv/creditcard.csv')
    index_c = data.Class.value_counts().index
    values_c = data.Class.value_counts().values
    # 提取Class种类和数量
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置图像中文字符，宋体
    fig, axes = plt.subplots()
    vert_bar = axes.bar(index_c, values_c, color='lightblue', align='center', width=0.5)
    axes.set(title='正常/欺诈数据比', ylabel='数据量', xlabel='0正常，1欺诈',
             xticks=index_c, xticklabels=index_c)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    # 去除上和右的边框
    for a, b in zip(index_c, values_c):
        axes.text(a, b+10000, b, ha='center', va='bottom', fontsize=11)
    # 添加条形上方的数据标签
    plt.savefig('../img/class_info.png')


def roc():
    error, labels, scores = test(4)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    # 获取roc函数的fpr、tpr和阈值

    fig, axes = plt.subplots()
    plt.plot(fpr, tpr, color='#FFFF00', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='#76EEC6', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('False Positive Rate,假正类率')
    plt.ylabel('True Positive Rate,真正类率')
    # 设置x、y轴坐标和标签
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # 设置图例的显示位置，为右下
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.savefig('../img/roc.png')


if __name__ == '__main__':
    roc()
