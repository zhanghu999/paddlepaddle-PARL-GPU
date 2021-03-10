import numpy as np
import pandas as pd


class Fcfs_Order:
    def __init__(self, df):
        CAPACITY = 100
        self.df = df  # 数据文件
        self.current_step = 0  # 当前步数
        self.current_capacity = CAPACITY  # 每一个阶段有300个产能
        self.need_capacity = 0  # 当前时间下完成所有已接受订单所需要的产能
        self.arrival_time_old = 1  #上一个订单的到达时间，若下一个订单与上一个订单一同达到，则完成当前订单所需要的产能不变


    def accept_order(self):
        arrival_time = self.df.iloc[self.current_step].arrival_time
        lead_time = self.df.iloc[self.current_step].lead_time
        delay_time = self.df.iloc[self.current_step].delay_time
        revenue = self.df.iloc[self.current_step].revenue

        if self.arrival_time_old != arrival_time:
            self.need_capacity -= (arrival_time - self.arrival_time_old) * 2
            if self.need_capacity < 0:
                self.current_capacity += self.need_capacity
                self.need_capacity = 0
        print("剩余产能=%d 完成当前已接受订单需要产能=%d " % (self.current_capacity, self.need_capacity))
        print("订单属性：到达时间%d 提前期%d 交货期%d 订单收益%d " % (arrival_time, lead_time, delay_time, revenue))
        if self.is_able_receive():
            reward = revenue
            self.current_capacity -= lead_time  # 当前剩余产能
            self.need_capacity = lead_time  # 完成当前订单所需要消耗的产能
            print(" \033[1;35m 接受订单 \033[0m")

        else:
            reward = 0
            print(" \033[1;33m 不满足接受要求的订单 \033[0m")
            print(" \033[1;32m 拒绝订单 \033[0m")

        self.current_step += 1  # 下一订单序列
        self.arrival_time_old = arrival_time

        print("剩余产能=%d 完成当前已接受订单需要产能=%d " % (self.current_capacity, self.need_capacity))
        return reward

    def is_able_receive(self):
        lead_time = self.df.iloc[self.current_step].lead_time
        delay_time = self.df.iloc[self.current_step].delay_time,
        if lead_time + self.need_capacity - delay_time > 0:
            return False  # 拒绝订单

        else:
            return True

    def is_done(self):
        ''' 判断当前是否已经结束 ，当产能使用完后则结束'''
        if self.current_capacity <= 0 or self.current_step > 149:
            done = False
        else:
            done = True
        return done

if __name__ == '__main__':
    df = pd.read_csv('data/data4.csv')
    Fcfs = Fcfs_Order(df)
    total_reward = 0
    while Fcfs.is_done():
        reward =Fcfs.accept_order()
        total_reward += reward
        print("累计收益=%d" %total_reward)
        print("-------------")


