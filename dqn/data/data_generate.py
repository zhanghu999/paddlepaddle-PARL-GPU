import numpy as np
import pandas as pd
import random
from numpy.random import choice

'''数组写入excel表'''
def excelWriter(A):
    data = pd.DataFrame(A)
    writer = pd.ExcelWriter('b.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f',index=None, header=None)  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()

def Arrive_time(num):
    time = [1]
    a = 1
    for i in range(0, num):
        n = choice([0, 1, 2],
                   p=[0.4, 0.3, 0.3])
        a += n
        time.append(a)
    time1 = np.array(time)
    return time1


def Lead_time():
    lead_time = choice([1, 2, 3],
                   size=150,
                   p=[0.5, 0.3, 0.2])
    return lead_time

def Delay_time():
    delay_time = choice([3, 4, 5, 6],
                   size=150,
                   p=[0.2, 0.3 ,0.3, 0.2])
    return delay_time

def Customer_level():
    customer_level = choice([1, 2, 3],
                     size=150,
                     p=[0.5, 0.3, 0.2])
    return customer_level




if __name__ == '__main__':
    arrive_time = Arrive_time(149)
    customer_level = Customer_level()
    delay_time = Delay_time()
    lead_time = Lead_time()
    order_data=np.transpose(np.vstack((arrive_time, customer_level,delay_time,lead_time)))
    excelWriter(order_data)




