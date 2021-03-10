import os
import pickle
import pandas as pd
from env import StockTradingEnv
from model import StockModel
from agent import StockAgent
from parl.algorithms import DDPG
from parl.utils import action_mapping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward


def stock_trade(stock_file,mode_path):
    day_profits = []
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')
    # The algorithms require a vectorized environment to run
    env = StockTradingEnv(df)

    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    # 使用parl框架搭建Agent：QuadrotorModel, DDPG, QuadrotorAgent三者嵌套
    model = StockModel(act_dim)
    algorithm = DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = StockAgent(algorithm, obs_dim, act_dim)

    df_test = pd.read_csv(stock_file.replace('train', 'test'))
    # 加载模型
    if os.path.exists(mode_path):
        agent.restore(mode_path)
    env2 = StockTradingEnv(df_test)
    obs = env2.reset()
    for i in range(len(df_test) - 1):
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        action = np.random.normal(action, 1.0)
        action = np.clip(action, -1.0, 1.0)  ## special
        action = action_mapping(action, env2.action_space.low[0],
                                env2.action_space.high[0])
        next_obs, reward, done, info = env2.step(action)

        obs = next_obs
        profit = env2.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits


def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade(stock_code,mode_path):
    stock_file = find_file('./stockdata/train', str(stock_code))

    daily_profits = stock_trade(stock_file,mode_path)
    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./{stock_code}.png')


def multi_stock_trade(mode_path):
    start_code = 600000
    max_num = 3000

    group_result = []

    for code in range(start_code, start_code + max_num):
        stock_file = find_file('./stockdata/train', str(code))
        if stock_file:
            try:
                profits = stock_trade(stock_file, mode_path)
                group_result.append(profits)
            except Exception as err:
                print(err)

    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


if __name__ == '__main__':
    # multi_stock_trade()
    test_a_stock_trade('sh.000001','model_dir/steps_30131.ckpt')
    # ret = find_file('./stockdata/train', '600036')
    # print(ret)