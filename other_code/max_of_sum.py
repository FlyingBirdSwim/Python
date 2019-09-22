#!usr/bin/env python
# coding=utf-8

# 输入：列表list
# 输出：list的子列表中，和最大的子列表（最少为一个元素）以及sum值
# 1.常规实现  2.尽量优化（未完成）
import time
from memory_profiler import profile


data = [-1, -2, -1, 1, -2, 3, -1, 2, -3, 2]


@profile()
def max_1(lt):
    max_sum = 0
    max_list = []
    for i in range(len(lt)):
        for j in range(len(lt) + 1):
            # list[i: j]遍历所有子列表，但是有较多空集
            if lt[i: j] == []:
                continue
            if sum(lt[i: j]) > max_sum:
                max_sum = sum(lt[i: j])
                max_list = lt[i: j]
    print('最大的和为 %d' % max_sum + '，列表为 %s' % max_list)


if __name__ == '__main__':
    start = time.process_time()
    max_1(data)
    end = time.process_time()
    print('运行时间为 %f' % (end-start))
