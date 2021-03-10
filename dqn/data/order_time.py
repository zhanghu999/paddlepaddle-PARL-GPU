import numpy as np
import pandas as pd
import random
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import scipy.stats as ss

'''数组写入excel表'''
def excelWriter(A):
    data = pd.DataFrame(A)
    writer = pd.ExcelWriter('B.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f',index=None, header=None)  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()


'''生成订单到达时间的随机序列'''
def order_time(num):
    time = [1]
    a = 1
    for i in range(0, num):
        n = random.randint(0, 2)
        a += n
        time.append(a)
    return time


'''生成每笔订单消耗的产能'''
def capatity_consume():
    cap = [1]
    for i in range(0, 100):
        a = random.randint(1, 3)
        cap.append(a)
    return cap


if __name__ == '__main__':
    A = order_time(120)
    excelWriter(A)
