#!/usr/bin/env python

# 输入：字符串类型的日期数据
# 输入：该日期是这一年的第几天
# 1.使用日期函数 2.不使用日期函数
import datetime as dt
import time
from memory_profiler import profile

# date = '2019-12-06'
# date = '2020-05-13'
date = '2020-01-21'


# 方法1
# @profile
def day_1(date):
    year = date[0: 4]
    d1 = dt.datetime.strptime(date, '%Y-%m-%d')
    d2 = dt.datetime.strptime(year + '-01-01',  '%Y-%m-%d')   # 别把strptime和strftime搞混
    delta = d1 - d2
    tm_day = delta.days + 1    # 第几天要加1天
    print('该日期是那一年的第 ' + str(tm_day) + ' 天')


# 方法2
# @profile
def day_2(date):
    year = int(date[0: 4])
    month = date[5: 7]
    day = int(date[8:])

    dict = {'01': 0, '02': 31, '03': 59, '04': 90, '05': 120, '06': 151,
            '07': 181, '08': 212, '09': 243, '10': 273, '11': 304, '12': 334}
    tm_day = dict.get(month) + day

    if year % 4 == 0 and dict.get(month) >= 59:
        tm_day += 1
    print('该日期是那一年的第 ' + str(tm_day) + ' 天')


if __name__ == '__main__':
    start = time.process_time()
    day_2(date)
    # day_1(date)
    end = time.process_time()
    print('运行时间为：', end - start)
# 方法2的运行时间更短，占用内存更少

