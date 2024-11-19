import sys
sys.path.append("/data/yueqi/Long_term_round_based_ridehailing_full")
import time
from setting import PKL_PATH
from s4_dqn_epsilon_decay import agents,modules,replay_buffers,explorers
import gym
import torch
import xs_gym


"""
s  该轮中订单数、空闲司机数、司机之间的收入方差、平均数、jain、当前轮次T  [](20,50,20,10,90)
"""

class TrainManager():

    def __init__(self,
                 env,  # 环境
                 episodes=700,  # 轮次数量
                 batch_size=32,  # 每一批次的数量
                 num_steps=4,  # 进行学习的频次
                 memory_size=2000,  # 经验回放池的容量
                 replay_start_size=200,  # 开始回放的次数
                 update_target_steps=200,  # 同步参数的次数
                 lr=0.001,  # 学习率
                 gamma=0.9,  # 收益衰减率
                 e_greed=0.08585299999998586,  # 探索与利用中的探索概率
                 e_gredd_decay=1e-6  # 探索与利用中探索概率的衰减步长
                 ):

        n_act = env.action_space.n    #  这个是动作的个数？
        n_obs = env.observation_space.shape[0]  #  这个是状态的维度？

        self.env = env
        self.episodes = episodes


        explorer = explorers.EpsilonGreedy(n_act,e_greed,e_gredd_decay)
        q_func = modules.MLP(n_obs, n_act)
        # q_func = torch.load("1500_09_every_100"+".pkl")
        # print("q_func = ","1500_09_every_100"+".pkl")
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = replay_buffers.ReplayBuffer(memory_size, num_steps)

        self.agent = agents.DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            explorer=explorer,
            replay_buffer = rb,
            batch_size=batch_size,
            replay_start_size = replay_start_size,
            update_target_steps = update_target_steps,
            n_act=n_act,
            gamma=gamma)

    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            if done: break
        print('e_greedy =',self.agent.explorer.epsilon)
        return total_reward

    def test_episode(self,is_render=False):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            obs = next_obs
            if is_render:self.env.render()
            if done: break
        return total_reward

    def test(self,is_render=False):
        total_reward = 0
        obs = self.env.reset()
        a = [10, 8, 10, 10, 10, 10, 4, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        cnt = 0
        while True:
            action = a[cnt]-1
            cnt += 1
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            obs = next_obs
            if is_render:self.env.render()
            if done: break
        print(
            '*********************  reward = %.1f **************************' % (
            total_reward))
        return total_reward

    def train(self):
        for e in range(self.episodes):
            ep_reward = self.train_episode()
            print('******************************************  Episode %s: reward = %.1f ********************************************' % (e, ep_reward))
            # if e % 100 == 0:
            #     test_reward = self.test_episode(False)
            #     print('test reward = %.1f' % (test_reward))
            if e > 0 and e % 100 == 0:
                PATH = "1500_09_every_100_1.pkl"
                torch.save(self.agent.pred_func, PATH)

        # 进行最后的测试
        PATH = "linux_1500_09_1000.pkl"  # result_09  lamuda = 0.9   result_09_time  时间换成了轮次数
        torch.save(self.agent.pred_func, PATH)
        # total_test_reward = 0
        # for i in range(5):
        #     total_test_reward += self.test_episode(False)
        # print('final test reward = %.1f' % (total_test_reward/5))

    def test_load_episode(self, is_render=False):
        print("test_load_episode")
        total_reward = 0
        obs = self.env.reset()

        while True:
            action = self.agent.load_predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            obs = next_obs
            if is_render: self.env.render()
            if done:
                self.env.close()
                break
        print("total_reward=",  total_reward)
        return total_reward


if __name__ == '__main__':
    env1 = gym.make("myEnv-v0")
    # env1 = gym.make("CartPole-v0")
    tm = TrainManager(env1)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # print("2000辆车，2000_09_every_100 l = 0 700轮继续训练，linux_1500_09_1000.pkl，l = 0")
    # tm.train()
    tm.test_load_episode()
    # tm.test()


