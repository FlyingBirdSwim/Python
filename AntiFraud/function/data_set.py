import pymysql as msl
import pandas as pd
import numpy as np


def make_table_sql(df):
    title = df.columns.tolist()
    # 读取属性名称
    types = df.ftypes
    # 获取每列属性类型
    make_table = []
    for item in title:
        if 'int' in types[item]:
            char = item + ' INT'
        elif 'float' in types[item]:
            char = item + ' FLOAT'
        elif 'object' in types[item]:
            char = item + ' VARCHAR(255)'
        elif 'datetime' in types[item]:
            char = item + ' DATETIME'
        # 注意空格隔开char中item和types
        make_table.append(char)
    return ','.join(make_table)
    # 给make_table加上，表示间隔
# 将csv文件的首行属性转化为sql表格属性，参数为dataframe


def save_to_mysql(base_name, table_name, df):
    conn = msl.connect('localhost', 'root', 'lxy318151')
    # 连接mysql
    conn.autocommit(1)
    # 确认连接
    cr = conn.cursor()
    # 创建游标
    conn.select_db(base_name)
    # 连接数据库

    cr.execute('DROP TABLE IF EXISTS {}'.format(table_name))
    cr.execute('CREATE TABLE IF NOT EXISTS {}({})'.format(table_name, make_table_sql(df)))
    # 删除已有同名table，创建新的table
    # format格式化函数，参数对应{}插入
    values = df.values.tolist()
    s = ','.join(['%s' for _ in range(len(df.columns))])
    # 设定插入格式
    cr.executemany('INSERT INTO {} VALUES ({})'.format(table_name, s), values)
    cr.execute('ALTER TABLE {} ADD Id INT AUTO_INCREMENT PRIMARY KEY'.format(table_name))
    # 批量插入数据
    conn.close()
# 插入数据库，参数为数据库名称，表名称，dataframe


def save_dataset():
    data = pd.read_csv('../csv/creditcard.csv')
    print('--已读取csv文件--')
    data = data.sample(frac=1)
    # 将数据集乱序
    data_1 = data[data.Class == 0]
    data_2 = data[data.Class == 1]
    # 分割出正常数据和欺诈数据
    a = data_2.shape[0]*2
    b = int(0.7 * len(data))
    data_3 = data_2.append(data_1[0:a], ignore_index=True)
    data_3 = data_3.sample(frac=1)
    data_4 = data[a: b+a]
    # 分割出训练集和测试集

    basename = 'creditcard'
    # save_to_mysql(basename, 'normaldata', data_1)
    # save_to_mysql(basename, 'frauddata', data_2)
    save_to_mysql(basename, 'traindata', data_3)
    print('--训练集录入完毕--')
    save_to_mysql(basename, 'testdata', data_4)
    print('--测试集录入完毕--')
# 将csv文件存储入数据库，并划分表


def load_dataset_to_sql(data_name):
    conn = msl.connect('localhost', 'root', 'lxy318151', 'creditcard')
    # 连接mysql
    cr = conn.cursor()
    # 创建游标
    cr.execute('SELECT * FROM {}'.format(data_name))
    data_set = cr.fetchall()
    labels_list = [i[30] for i in data_set]
    data_list = [i[1:29] for i in data_set]
    return data_list, labels_list
# 获取数据库表数据，返回v1至v28列的data列表和class列的label列表


def cut_dataset():
    data = pd.read_csv('../csv/creditcard.csv')
    data = data.sample(frac=1)
    data_1 = data[data.Class == 0]
    data_2 = data[data.Class == 1]
    # 分割出正常数据和欺诈数据
    a = data_2.shape[0]*2
    b = int(0.7 * len(data))
    data_3 = data_2.append(data_1[0:a], ignore_index=True)       # 测试集正常数据和（所有的）欺诈数据2:1比例
    data_3 = data_3.sample(frac=1)
    data_4 = data[a: b+a]         # 测试集

    data_3.to_csv('../csv/traindata.csv')
    data_4.to_csv('../csv/testdata.csv')
    print('划分完毕')
# 不使用mysql数据库，直接划分训练集、测试集存储为csv


def load_dataset_to_csv(name):
    path = '../csv/' + name + '.csv'
    data_set = pd.read_csv(path)
    data_set_list = np.array(data_set).tolist()
    # print(data_set_list)
    data_list = [i[2:30] for i in data_set_list]
    labels_list = [i[31] for i in data_set_list]
    return data_list, labels_list
# csv读取数据和标签


if __name__ == '__main__':
    # save_dataset()
    cut_dataset()

