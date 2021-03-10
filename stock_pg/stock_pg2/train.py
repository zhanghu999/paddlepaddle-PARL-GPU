import os
import gym
import numpy as np
import parl
from parl.utils import logger  # 日志打印工具
import matplotlib.pyplot as plt
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

from model import Model
from algorithm import PolicyGradient  # from parl.algorithms import PolicyGradient  # parl >= 1.3.1
from agent import Agent

LEARNING_RATE = 1e-4 #学习率

def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = data_process(obs)
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list

# 评估 agent, 跑 1 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(1):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = data_process(obs)
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

# 数据预处理
def data_process(obs):
    # 缩放数据尺度
    obs[:,0]/=100
    return obs.astype(np.float).ravel()


# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.9):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    #reward_arr -= np.mean(reward_arr)
    #reward_arr /= np.std(reward_arr)
    return reward_arr

def main():
    # 创建环境

    env_test = gym.make('stocks-v0', frame_bound=(1800, 2150), window_size=10)
    obs_dim = 20
    act_dim = 2
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # 加载模型
    # agent.restore('./stock_pg_v1_2.ckpt')

    test_flag = 1  # 是否直接测试

    if (test_flag == 1):
        for i in range(5000):

            # 每次跟新训练环境
            start = np.random.randint(10, 1900)
            env_train = gym.make('stocks-v0', frame_bound=(start, start + 100), window_size=10)

            # 每次都是单个环境
            # env_train = gym.make('stocks-v0', frame_bound=(10, 2000), window_size=10)

            obs_list, action_list, reward_list = run_episode(env_train, agent)

            if i % 50 == 0:
                logger.info("Train Episode {}, Reward Sum {}.".format(i,
                                                                      sum(reward_list)))

            batch_obs = np.array(obs_list)
            batch_action = np.array(action_list)
            batch_reward = calc_reward_to_go(reward_list)

            cost = agent.learn(batch_obs, batch_action, batch_reward)
            if (i + 1) % 100 == 0:
                total_reward = evaluate(env_test, agent)
                logger.info('Episode {}, Test reward: {}'.format(i + 1,
                                                                 total_reward))
                # 保存模型
                ckpt = 'stock_pg_v2/steps_{}.ckpt'.format(i)
                agent.save(ckpt)

                plt.cla()
                env_test.render_all()
                plt.show()

        # save the parameters to ./model.ckpt
        # agent.save('./stock_pg_v1_2.ckpt')

    else:
        # 加载模型
        agent.restore('./stock_pg_v2/steps_4899.ckpt')
        total_reward = evaluate(env_test, agent, render=True)
        logger.info('Test reward: {}'.format(total_reward))
        plt.cla()
        env_test.render_all()
        plt.show()

if __name__ == '__main__':
    main()
