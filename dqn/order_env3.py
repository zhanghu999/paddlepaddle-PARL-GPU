import gym
import pandas as pd
from gym import spaces
import numpy as np


class CustomEnv(gym.Env):
    # metadata = {'render.modes' : ['human']}
    def __init__(self, df):
        self.df = df
        self.env = Order(df)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0]),
                                            np.array([152, 10, 10, 10, 10, 10, 10]),
                                            dtype=np.int)

    def reset(self, df):
        del self.env
        self.env = Order(df)
        return self.env.observe()

    def step(self, action):
        return self.env.take_action(action)

    def render(self, mode="human", close=False):
        self.env.view()


class Order:
    def __init__(self, df):
        CAPACITY = 100
        self.df = df  # 数据文件
        self.current_step = 0  # 当前步数
        self.current_capacity = CAPACITY  # 每一个阶段有300个产能
        self.need_capacity = 0  # 当前时间下完成所有已接受订单所需要的产能
        self.arrival_time_old = 1  # 上一个订单的到达时间，若下一个订单与上一个订单一同达到，则完成当前订单所需要的产能不变

    def observe(self):
        obs = np.array([
            self.df.iloc[self.current_step].arrival_time,
            self.df.iloc[self.current_step].customer_level,
            self.df.iloc[self.current_step].delay_time,
            self.df.iloc[self.current_step].lead_time,
            self.df.iloc[self.current_step].daily_capacity,
            self.df.iloc[self.current_step].revenue,
            self.need_capacity,
        ])
        return obs  # 返回当前订单的状态

    def take_action(self, action):

        arrival_time = self.df.iloc[self.current_step].arrival_time
        customer_level = self.df.iloc[self.current_step].customer_level
        lead_time = self.df.iloc[self.current_step].lead_time
        delay_time = self.df.iloc[self.current_step].delay_time
        daily_capacity = self.df.iloc[self.current_step].daily_capacity
        revenue = self.df.iloc[self.current_step].revenue

        if self.arrival_time_old != arrival_time:
            self.need_capacity -= (arrival_time - self.arrival_time_old) * 2
            if self.need_capacity < 0:
                self.current_capacity += self.need_capacity
                self.need_capacity = 0

        print("订单决策前：剩余产能=%d 完成当前已接受订单需要产能=%d " % (self.current_capacity, self.need_capacity))
        print("订单属性：到达时间%d 提前期%d 交货期%d 订单收益%d " % (arrival_time, lead_time, delay_time, revenue))

        if self.is_able_receive():
            action = 1
            print(" \033[1;33m 不满足接受要求的订单 \033[0m")

        if action == 0:  # 接受订单
            # reward = revenue + customer_level*2
            reward = revenue
            self.current_capacity -= lead_time  # 当前剩余产能
            self.need_capacity = lead_time  # 完成当前订单所需要消耗的产能
            print(" \033[1;35m 接受订单 \033[0m")

        if action == 1:  # 拒绝订单
            reward = 0
            print(" \033[1;32m 拒绝订单 \033[0m")

        self.current_step += 1  # 下一订单序列
        self.arrival_time_old = arrival_time
        print("订单决策后：剩余产能=%d 完成当前已接受订单需要产能=%d " % (self.current_capacity, self.need_capacity))
        done = self.is_done()
        obs = np.array(
            [arrival_time, customer_level, delay_time, lead_time, daily_capacity, revenue, self.need_capacity])
        return obs, reward, done, {}

    def evaluate(self):
        pass

    def is_done(self):
        ''' 判断当前是否已经结束 ，当产能使用完后则结束'''
        if self.current_capacity <= 0 or self.current_step > 149:
            done = True
        else:
            done = False
        return done

    def is_able_receive(self):
        lead_time = self.df.iloc[self.current_step].lead_time
        delay_time = self.df.iloc[self.current_step].delay_time,
        if lead_time + self.need_capacity - delay_time > 0:
            return True  # 拒绝订单
        else:
            return False

    def view(self):
        pass


if __name__ == '__main__':
    NUM = 100  # 循环次数
    df = pd.read_csv('data/data4.csv')
    env = CustomEnv(df)
    total_reward = 0
    total_total_reward = 0
    for i in range(NUM):
        env.reset(df)
        for t in range(200):
            action = env.action_space.sample()
            observation, r, done, info = env.step(action)
            total_reward += r
            print("累计收益=%d" % total_reward)
            print("-------------")
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                total_total_reward += total_reward
                total_reward = 0
                break
    average_total_reward = total_total_reward/NUM
    print("平均累计收益",average_total_reward)

