"""
Vanilla Policy Gradient(VPG or REINFORCE)
-----------------------------------------
The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance.
It's an on-policy algorithm can be used for environments with either discrete or continuous action spaces.
Here is an example on discrete action space game CartPole-v0.
To apply it on continuous action space, you need to change the last softmax layer and the choose_action function.

Reference
---------
Cookbook: Barto A G, Sutton R S. Reinforcement Learning: An Introduction[J]. 1998.
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
-----------
Openai Gym CartPole-v0, discrete action space

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_PG.py --train/test

"""
import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'CartPole-v0'    # 定义环境
RANDOMSEED = 1              # 设置随机种子。建议大家都设置，这样试验可以重现。

DISPLAY_REWARD_THRESHOLD = 400  # 如果奖励超过DISPLAY_REWARD_THRESHOLD，就开始渲染
RENDER = False                  # 开始的时候，不渲染游戏。
num_episodes = 2                # 游戏迭代次数

###############################  PG  ####################################


class PolicyGradient:
    """
    PG class
    """

    def __init__(self, n_features, n_actions, learning_rate=0.01, reward_decay=0.95):
        # 定义相关参数
        self.n_actions = n_actions      #动作
        self.n_features = n_features    #环境特征数量
        self.lr = learning_rate         #学习率
        self.gamma = reward_decay       #折扣

        #用于保存每个ep的数据。
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        def get_model(inputs_shape):
            """
            创建一个神经网络
            输入: state
            输出: act
            """
            with tf.name_scope('inputs'):
                self.tf_obs = tl.layers.Input(inputs_shape, tf.float32, name="observations")
                #self.tf_acts = tl.layers.Input([None,], tf.int32, name="actions_num")
                #self.tf_vt = tl.layers.Input([None,], tf.float32, name="actions_value")
            # fc1
            layer = tl.layers.Dense(
                n_units=30, act=tf.nn.tanh, W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
                b_init=tf.constant_initializer(0.1), name='fc1'
            )(self.tf_obs)
            # fc2
            all_act = tl.layers.Dense(
                n_units=self.n_actions, act=None, W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
                b_init=tf.constant_initializer(0.1), name='all_act'
            )(layer)
            return tl.models.Model(inputs=self.tf_obs, outputs=all_act, name='PG model')

        self.model = get_model([None, n_features])
        self.model.train()
        self.optimizer = tf.optimizers.Adam(self.lr)

    def choose_action(self, s):
        """
        用神经网络输出的**策略pi**，选择动作。
        输入: state
        输出: act
        """
        _logits = self.model(np.array([s], np.float32))     
        _probs = tf.nn.softmax(_logits).numpy()             
        return tl.rein.choice_action_by_probs(_probs.ravel())   #根据策略PI选择动作。

    def choose_action_greedy(self, s):
        """
        贪心算法：直接用概率最大的动作
        输入: state
        输出: act
        """
        _probs = tf.nn.softmax(self.model(np.array([s], np.float32))).numpy()
        return np.argmax(_probs.ravel())

    def store_transition(self, s, a, r):
        """
        保存数据到buffer中
        """
        self.ep_obs.append(np.array([s], np.float32))
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        """
        通过带权重更新方法更新神经网络
        """
        # _discount_and_norm_rewards中存储的就是这一ep中，每个状态的G值。
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        with tf.GradientTape() as tape:
            
            # 把s放入神经网络，就算_logits
            _logits = self.model(np.vstack(self.ep_obs))
            
            # 敲黑板
            ## _logits和真正的动作的差距
            # 差距也可以这样算,和sparse_softmax_cross_entropy_with_logits等价的:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=np.array(self.ep_as))

            # 在原来的差距乘以G值，也就是以G值作为更新
            loss = tf.reduce_mean(neg_log_prob * discounted_ep_rs_norm)

        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        """
        通过回溯计算G值
        """
        # 先创建一个数组，大小和ep_rs一样。ep_rs记录的是每个状态的收获r。
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 从ep_rs的最后往前，逐个计算G
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 归一化G值。
        # 我们希望G值有正有负，这样比较容易学习。
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/pg_policy.hdf5', self.model)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/pg_policy.hdf5', self.model)


if __name__ == '__main__':

    # reproducible
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    tl.logging.set_verbosity(tl.logging.DEBUG)

    env = gym.make(ENV_NAME)
    env.seed(RANDOMSEED)  # reproducible, general Policy gradient has high variance
    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.99,
        # output_graph=True,
    )

    if args.train:
        reward_buffer = []

        #=====开始更新训练=====
        for i_episode in range(num_episodes):

            episode_time = time.time()
            observation = env.reset()

            while True:
                if RENDER:
                    env.render()

                # 注意：这里没有用贪婪算法，而是根据pi随机动作，以保证一定的探索性。
                action = RL.choose_action(observation)

                observation_, reward, done, info = env.step(action)

                # 保存数据
                RL.store_transition(observation, action, reward)

                # PG用的是MC，如果到了最终状态
                if done:
                    ep_rs_sum = sum(RL.ep_rs)

                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                    #如果超过DISPLAY_REWARD_THRESHOLD就开始渲染游戏吧。
                    if running_reward > DISPLAY_REWARD_THRESHOLD:
                        RENDER = True 

                    # print("episode:", i_episode, "  reward:", int(running_reward))

                    print(
                        "Episode [%d/%d] \tsum reward: %d  \trunning reward: %f \ttook: %.5fs " %
                        (i_episode, num_episodes, ep_rs_sum, running_reward, time.time() - episode_time)
                    )
                    reward_buffer.append(running_reward)

                    # 开始学习
                    vt = RL.learn()

                    # 画图
                    plt.ion()
                    plt.cla()
                    plt.title('PG')
                    plt.plot(reward_buffer, )  
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                    plt.pause(0.1)

                    break
                
                # 开始新一步
                observation = observation_
        RL.save_ckpt()
        plt.ioff()
        plt.show()

    # =====test=====
    RL.load_ckpt()
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)      # 这里建议大家可以改贪婪算法获取动作，对比效果是否有不同。
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
