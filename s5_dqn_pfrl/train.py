import sys
sys.path.append("/data/yueqi/Long_term_round_based_ridehailing_full")
from setting import PKL_PATH


import time

import gym
import torch
import pfrl
import numpy
from s5_dqn_pfrl import modules
import xs_gym

class TrainManager():

    def __init__(self,
                 is_train,
                env,#环境
                episodes=1000,#轮次数量
                batch_size = 32,#每一批次的数量
                num_steps=4,#进行学习的频次
                memory_size = 2000,#经验回放池的容量
                replay_start_size = 200,#开始回放的次数
                update_target_steps = 200,#同步参数的次数
                lr=0.001,#学习率
                gamma=0.9, #收益衰减率
                e_greed=0.1, #探索与利用中的探索概率
                #e_gredd_decay = 1e-6 #探索与利用中探索概率的衰减步长
                ):

        n_act = env.action_space.n
        n_obs = env.observation_space.shape[0]

        self.env = env
        self.episodes = episodes
        print("episodes = ",episodes)

        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=e_greed, random_action_func=env.action_space.sample)
        q_func = modules.MLP(n_obs, n_act)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = pfrl.replay_buffers.ReplayBuffer(capacity=memory_size,num_steps=num_steps)
        self.agent = pfrl.agents.DQN(
            q_function=q_func,
            optimizer=optimizer,
            explorer=explorer,
            replay_buffer=rb,
            minibatch_size=batch_size,
            replay_start_size=replay_start_size,
            target_update_interval=update_target_steps,
            gamma=gamma,
            phi=lambda x: numpy.array(x, dtype=numpy.float32)
        )

        if not is_train:
            print("PKL_PATH = ",PKL_PATH)
            self.agent.load("result/"+PKL_PATH)

    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action = self.agent.act(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.agent.observe(obs, reward, done, done)
            if done: break
        return total_reward

    def test(self,is_render=False):
        total_reward = 0
        for i in range(5):
            total_reward += self.test_episode(is_render)
        print("total_reward = ", total_reward / 5)
    def test_episode(self,is_render=False):
        with self.agent.eval_mode():
            obs = self.env.reset()
            total_reward = 0
            while True:
                action = self.agent.act(obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                if is_render:self.env.render()
                if done: break
        print("total_reward = ", total_reward)
        return total_reward

    def train(self):
        max_reward = 0
        for e in range(self.episodes):
            ep_reward = self.train_episode()
            print('************************Episode %s: reward = %.1f***************************' % (e, ep_reward))
            if max_reward < ep_reward:
                max_reward = ep_reward
                self.agent.save("result/best_2500_09_10")
            if e > 0 and e % 100 == 0:
                self.agent.save("result/every_100_2500_10per")

        # 进行最后的测试
        PATH = "result/train_09_2500_1000ep_10per"  # result_09  lamuda = 0.85
        self.agent.save(PATH)
        # total_test_reward = 0
        # for i in range(5):
        #     total_test_reward += self.test_episode(False)
        # print('final test reward = %.1f' % (total_test_reward/5))

if __name__ == '__main__':
    env1 = gym.make("myEnv-v0")
    # env1 = gym.make("CartPole-v0")
    tm = TrainManager(is_train=False,env=env1)
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # print("2500辆车，lamuda = 0.9，best_2500_09_10, every_100_2500_10per,train_09_2500_1000ep_10per,l = 0,1,取司机最后收入百分之10的和")
    # tm.train()
    tm.test_episode()


