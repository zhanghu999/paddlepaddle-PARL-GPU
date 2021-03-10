import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing

class BitcoinTradingEnv(gym.Env):
    '''
    接下来，让我们为环境创建类。
    我们将需要传入一个  pandas dataframe，以及一个可选的  initial_balance  和一个 lookback_window_size ，它将指示agent在过去的每一步将观察多少时间步长。
    我们将把每笔交易的佣金默认为0.075%，即Bitmex的当前利率，把  serial  参数默认为false，这意味着我们的dataframe将在默认情况下以随机切片的形式遍历。
    我们还在数据帧上调用 dropna()  和  reset_index()  来首先删除任何带有  NaN  值的行，
    然后重置frame的索引，因为我们已经删除了数据。
    '''
    metadata = {'render.modes': ['live', 'file', 'none']}