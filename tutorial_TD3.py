'''
Twin Delayed DDPG (TD3)
------------------------
DDPG suffers from problems like overestimate of Q-values and sensitivity to hyper-parameters.
Twin Delayed DDPG (TD3) is a variant of DDPG with several tricks:
* Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), 
and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.

* Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently 
than the Q-function. 

* Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for 
the policy to exploit Q-function errors by smoothing out Q along changes in action.

The implementation of TD3 includes 6 networks: 2 Q-net, 2 target Q-net, 1 policy net, 1 target policy net
Actor policy in TD3 is deterministic, with Gaussian exploration noise.

Reference
---------
original paper: https://arxiv.org/pdf/1802.09477.pdf


Environment
---
Openai Gym Pendulum-v0, continuous action space
https://gym.openai.com/envs/Pendulum-v0/

Prerequisites
---
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

&&
pip install box2d box2d-kengz --user

To run
-------
python tutorial_TD3.py --train/test

'''

import argparse
import math
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import clear_output

import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model

tfd = tfp.distributions
Normal = tfd.Normal

tl.logging.set_verbosity(tl.logging.DEBUG)

random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)  # reproducible

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################
# choose env
ENV = 'Pendulum-v0'
action_range = 1.  # scale action, [-action_range, action_range]

# RL training
max_frames = 40000  # total number of steps for training
test_frames = 300  # total number of steps for testing
max_steps = 150  # maximum number of steps for one episode
batch_size = 64  # udpate batchsize
explore_steps = 500  # 500 for random action sampling in the beginning of training
update_itr = 3  # repeated updates for single step
hidden_dim = 32  # size of hidden layers for networks
q_lr = 3e-4  # q_net learning rate
policy_lr = 3e-4  # policy_net learning rate
policy_target_update_interval = 3  # delayed steps for updating the policy network and target networks
explore_noise_scale = 1.0  # range of action noise for exploration
eval_noise_scale = 0.5  # range of action noise for evaluation of action value
reward_scale = 1.  # value range of reward
replay_buffer_size = 5e5  # size of replay buffer

###############################  TD3  ####################################


class ReplayBuffer:
    '''
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    '''

    def __init__(self, capacity):
        self.capacity = capacity        #buffer的最大值
        self.buffer = []                #buffer列表
        self.position = 0               #当前输入的位置，相当于指针

    def push(self, state, action, reward, next_state, done):
        #如果buffer的长度小于最大值，也就是说，第一环的时候，需要先初始化一个“空间”，这个空间值为None，再给这个空间赋值。
        if len(self.buffer) < self.capacity:        
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

#在代码中没有用到，但我们可以学习下，这里是直接修改gym环境的动作输出，把输出归一化。
class NormalizedActions(gym.ActionWrapper):
    ''' normalize the actions to be in reasonable range '''

    def _action(self, action):
        low = self.action_space.low                     #动作空间最小值
        high = self.action_space.high                   #动作空间最大值

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class QNetwork(Model):
    ''' the network for evaluate values of state-action pairs: Q(s,a) '''

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    ''' the network for generating non-determinstic (Gaussian distributed) action from the state input '''

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')

        self.output_linear = Dense(n_units=num_actions, W_init=w_init, \
        b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim, name='policy_output')

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        output = tf.nn.tanh(self.output_linear(x))  # unit range output [-1, 1]

        return output

    def evaluate(self, state, eval_noise_scale):
        ''' 
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        '''
        state = state.astype(np.float32)        #状态的type整理
        action = self.forward(state)            #通过state计算action，注意这里action范围是[-1,1]

        action = self.action_range * action     #映射到游戏的action取值范围

        # add noise
        normal = Normal(0, 1)                   #建立一个正态分布
        eval_noise_clip = 2 * eval_noise_scale  #对噪声进行上下限裁剪。eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale      #弄个一个noisy和action的shape一致，然后乘以scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)  #对noisy进行剪切，不要太大也不要太小
        action = action + noise                 #action加上噪音

        return action

    #输入state，输出action
    def get_action(self, state, explore_noise_scale):
        ''' generate action with state for interaction with envronment '''
        action = self.forward([state])          #这里的forward函数，就是输入state，然后通过state输出action。只不过形式不一样而已。最后的激活函数式tanh，所以范围是[-1, 1]
        action = action.numpy()[0]              #获得的action变成矩阵。

        # add noise
        normal = Normal(0, 1)                   #生成normal这样一个正态分布              
        noise = normal.sample(action.shape) * explore_noise_scale       #在正态分布中抽样一个和action一样shape的数据，然后乘以scale
        action = self.action_range * action + noise     #action乘以动作的范围，加上noise

        return action.numpy()

    def sample_action(self, ):
        ''' generate random actions for exploration '''
        a = tf.random.uniform([self.num_actions], -1, 1)

        return self.action_range * a.numpy()


class TD3_Trainer():

    def __init__(
            self, replay_buffer, hidden_dim, action_range, policy_target_update_interval=1, q_lr=3e-4, policy_lr=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        # 用两个Qnet来估算，doubleDQN的想法。同时也有两个对应的target_q_net
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        # 把net 赋值给target_network
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.update_cnt = 0     #更新次数
        self.policy_target_update_interval = policy_target_update_interval      #策略网络更新频率

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    #在网络初始化的时候进行硬更新
    def target_ini(self, net, target_net):
        ''' hard-copy update for initializing target networks '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    #在更新的时候进行软更新
    def target_soft_update(self, net, target_net, soft_tau):
        ''' soft update the target net with Polyak averaging '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
                # 原来参数占比 + 目前参数占比
            )
        return target_net

    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        ''' update all networks in TD3 '''
        self.update_cnt += 1        #计算更新次数
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)     #从buffer sample数据

        reward = reward[:, np.newaxis]  # expand dim， 调整形状，方便输入网络
        done = done[:, np.newaxis]

        # 输入s',从target_policy_net计算a'。注意这里有加noisy的
        new_next_action = self.target_policy_net.evaluate(
            next_state, eval_noise_scale=eval_noise_scale
        )  # clipped normal noise

        # 归一化reward.(有正有负)
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        # 把s'和a'堆叠在一起，一起输入到target_q_net。
        # 有两个qnet，我们取最小值
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        #计算target_q的值，用于更新q_net
        #之前有把done从布尔变量改为int，就是为了这里能够直接计算。
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # input of q_net

        #更新q_net1
        #这里其实和DQN是一样的
        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        #更新q_net2
        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        # policy不是经常updata的，而qnet更新一定次数，才updata一次
        if self.update_cnt % self.policy_target_update_interval == 0:
            #更新policy_net
            with tf.GradientTape() as p_tape:
                # 计算 action = Policy(s)，注意这里是没有noise的
                new_action = self.policy_net.evaluate(
                    state, eval_noise_scale=0.0
                )  # no noise, deterministic policy gradients

                #叠加state和action
                new_q_input = tf.concat([state, new_action], 1)
                # ''' implementation 1 '''
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                ''' implementation 2 '''
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)    #梯度上升
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            # 软更新target_network三个
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save_weights(self):  # save trained weights
        tl.files.save_npz(self.q_net1.trainable_weights, name='model_q_net1.npz')
        tl.files.save_npz(self.q_net2.trainable_weights, name='model_q_net2.npz')
        tl.files.save_npz(self.target_q_net1.trainable_weights, name='model_target_q_net1.npz')
        tl.files.save_npz(self.target_q_net2.trainable_weights, name='model_target_q_net2.npz')
        tl.files.save_npz(self.policy_net.trainable_weights, name='model_policy_net.npz')
        tl.files.save_npz(self.target_policy_net.trainable_weights, name='model_target_policy_net.npz')

    def load_weights(self):  # load trained weights
        tl.files.load_and_assign_npz(name='model_q_net1.npz', network=self.q_net1)
        tl.files.load_and_assign_npz(name='model_q_net2.npz', network=self.q_net2)
        tl.files.load_and_assign_npz(name='model_target_q_net1.npz', network=self.target_q_net1)
        tl.files.load_and_assign_npz(name='model_target_q_net2.npz', network=self.target_q_net2)
        tl.files.load_and_assign_npz(name='model_policy_net.npz', network=self.policy_net)
        tl.files.load_and_assign_npz(name='model_target_policy_net.npz', network=self.target_policy_net)


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig('td3.png')
    # plt.show()


if __name__ == '__main__':

    # initialization of env
    # env = NormalizedActions(gym.make(ENV))
    env = gym.make(ENV).unwrapped               #环境
    action_dim = env.action_space.shape[0]      #动作空间
    state_dim = env.observation_space.shape[0]  #状态空间

    # initialization of buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # initialization of trainer
    td3_trainer=TD3_Trainer(replay_buffer, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, \
    action_range=action_range, q_lr=q_lr, policy_lr=policy_lr )

    # set train mode
    td3_trainer.q_net1.train()
    td3_trainer.q_net2.train()
    td3_trainer.target_q_net1.train()
    td3_trainer.target_q_net2.train()
    td3_trainer.policy_net.train()
    td3_trainer.target_policy_net.train()

    # training loop
    if args.train:
        frame_idx = 0                           #总步数
        rewards = []                            #记录每个EP的总reward
        t0 = time.time()
        while frame_idx < max_frames:           #小于最大步数，就继续训练
            state = env.reset()                 #初始化state
            state = state.astype(np.float32)    #整理state的类型
            episode_reward = 0
            if frame_idx < 1:                   #第一次的时候，要进行初始化trainer
                print('intialize')
                _ = td3_trainer.policy_net([state])  # need an extra call here to make inside functions be able to use model.forward
                _ = td3_trainer.target_policy_net([state])

            # 开始训练
            for step in range(max_steps):
                if frame_idx > explore_steps:       #如果小于500步，就随机，如果大于就用get-action
                    action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)  #带有noisy的action
                else:
                    action = td3_trainer.policy_net.sample_action()

                # 与环境进行交互
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                env.render()
                done = 1 if done ==True else 0

                #记录数据在replay_buffer
                replay_buffer.push(state, action, reward, next_state, done)

                #赋值state，累计总reward，步数
                state = next_state
                episode_reward += reward
                frame_idx += 1

                #如果数据超过一个batch_size的大小，那么就开始更新
                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):         #注意：这里更新可以更新多次！
                        td3_trainer.update(batch_size, eval_noise_scale=0.5, reward_scale=1.)

                if frame_idx % 500 == 0:
                    plot(frame_idx, rewards)

                if done:
                    break
            episode = int(frame_idx / max_steps)  # current episode
            all_episodes = int(max_frames / max_steps)  # total episodes
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(episode, all_episodes, episode_reward, time.time()-t0 ))
            rewards.append(episode_reward)
        td3_trainer.save_weights()

    if args.test:
        frame_idx = 0
        rewards = []
        t0 = time.time()

        td3_trainer.load_weights()

        while frame_idx < test_frames:
            state = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            if frame_idx < 1:
                print('intialize')
                _ = td3_trainer.policy_net(
                    [state]
                )  # need an extra call to make inside functions be able to use forward
                _ = td3_trainer.target_policy_net([state])

            for step in range(max_steps):
                action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                env.render()
                done = 1 if done ==True else 0

                state = next_state
                episode_reward += reward
                frame_idx += 1

                # if frame_idx % 50 == 0:
                #     plot(frame_idx, rewards)

                if done:
                    break
            episode = int(frame_idx / max_steps)
            all_episodes = int(test_frames / max_steps)
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(episode, all_episodes, episode_reward, time.time()-t0 ) )
            rewards.append(episode_reward)
