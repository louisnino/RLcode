"""
Distributed Proximal Policy Optimization (DPPO)
----------------------------
A distributed version of OpenAI's Proximal Policy Optimization (PPO).
Workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

Reference
---------
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_DPPO.py --train/test


"""

import argparse
import os
import queue
import threading
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

GAME = 'Pendulum-v0'  # environment name
RANDOMSEED = 1  # random seed

EP_MAX = 1000  # total number of episodes for training
EP_LEN = 200  # total number of steps for each episode
GAMMA = 0.9  # reward discount
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
BATCH = 32  # update batchsize
A_UPDATE_STEPS = 10  # actor update steps
C_UPDATE_STEPS = 10  # critic update steps
S_DIM, A_DIM = 3, 1  # state dimension, action dimension
EPS = 1e-8  # epsilon

#PPO1 和PPO2 的参数
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization

N_WORKER = 4  # parallel workers
MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
UPDATE_STEP = 10  # loop update operation n-steps

###############################  DPPO  ####################################


class PPO(object):
    '''
    PPO class
    '''

    def __init__(self):

        # 创建critic
        tfs = tl.layers.Input([None, S_DIM], tf.float32, 'state')
        l1 = tl.layers.Dense(100, tf.nn.relu)(tfs)
        v = tl.layers.Dense(1)(l1)
        self.critic = tl.models.Model(tfs, v)
        self.critic.train()

        # 创建actor
        self.actor = self._build_anet('pi', trainable=True)
        self.actor_old = self._build_anet('oldpi', trainable=False)
        self.actor_opt = tf.optimizers.Adam(A_LR)
        self.critic_opt = tf.optimizers.Adam(C_LR)

    # 更新actor
    def a_train(self, tfs, tfa, tfadv):
        '''
        Update policy network
        :param tfs: state
        :param tfa: act
        :param tfadv: advantage
        :return:
        '''
        tfs = np.array(tfs, np.float32)
        tfa = np.array(tfa, np.float32)
        tfadv = np.array(tfadv, np.float32)     #td-error
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(tfs)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(tfs)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + EPS)
            surr = ratio * tfadv

            ## PPO1
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            ## PPO2
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * tfadv)
                )
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)

        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    # 更新old_pi
    def update_old_pi(self):
        '''
        Update old policy parameter
        :return: None
        '''
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)

    # 更新critic
    def c_train(self, tfdc_r, s):
        '''
        Update actor network
        :param tfdc_r: cumulative reward
        :param s: state
        :return: None
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = tfdc_r - self.critic(s)     #计算advantage：V(s') * gamma + r - V(s)
            closs = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    # 计算advantage：V(s') * gamma + r - V(s)
    def cal_adv(self, tfs, tfdc_r):
        '''
        Calculate advantage
        :param tfs: state
        :param tfdc_r: cumulative reward
        :return: advantage
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)
        return advantage.numpy()

    def update(self):
        '''
        Update parameter with the constraint of KL divergent
        :return: None
        '''
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():              #如果协调器没有停止
            if GLOBAL_EP < EP_MAX:                  #EP_MAX是最大更新次数
                UPDATE_EVENT.wait()                 #PPO进程的等待位置
                self.update_old_pi()                # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)

                s, a, r = data[:, :S_DIM].astype(np.float32), \
                          data[:, S_DIM: S_DIM + A_DIM].astype(np.float32), \
                          data[:, -1:].astype(np.float32)

                adv = self.cal_adv(s, r)
                # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

                # update actor
                ## PPO1
                if METHOD['name'] == 'kl_pen':
                    for _ in range(A_UPDATE_STEPS):
                        kl = self.a_train(s, a, adv)
                        if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                            break
                    if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                        METHOD['lam'] /= 2
                    elif kl > METHOD['kl_target'] * 1.5:
                        METHOD['lam'] *= 2
                    # sometimes explode, this clipping is MorvanZhou's solution
                    METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)

                ## PPO2
                else:  # clipping method, find this is better (OpenAI's paper)
                    for _ in range(A_UPDATE_STEPS):
                        self.a_train(s, a, adv)

                # update critic
                for _ in range(C_UPDATE_STEPS):
                    self.c_train(r, s)

                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    #build actor network
    def _build_anet(self, name, trainable):
        '''
        Build policy network
        :param name: name
        :param trainable: trainable flag
        :return: policy network
        '''
        tfs = tl.layers.Input([None, S_DIM], tf.float32, name + '_state')
        l1 = tl.layers.Dense(100, tf.nn.relu, name=name + '_l1')(tfs)
        a = tl.layers.Dense(A_DIM, tf.nn.tanh, name=name + '_a')(l1)
        mu = tl.layers.Lambda(lambda x: x * 2, name=name + '_lambda')(a)
        sigma = tl.layers.Dense(A_DIM, tf.nn.softplus, name=name + '_sigma')(l1)
        model = tl.models.Model(tfs, [mu, sigma], name)

        if trainable:
            model.train()
        else:
            model.eval()
        return model

    # 选择动作
    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        s = s[np.newaxis, :].astype(np.float32)
        mu, sigma = self.actor(s)
        pi = tfp.distributions.Normal(mu, sigma)
        a = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
        return np.clip(a, -2, 2)

    #计算V()
    def get_v(self, s):
        '''
        Compute value
        :param s: state
        :return: value
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/dppo_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/dppo_actor_old.hdf5', self.actor_old)
        tl.files.save_weights_to_hdf5('model/dppo_critic.hdf5', self.critic)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/dppo_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/dppo_actor_old.hdf5', self.actor_old)
        tl.files.load_hdf5_to_weights_in_order('model/dppo_critic.hdf5', self.critic)


'''--------------------------------------------------------------'''


class Worker(object):
    '''
    Worker class for distributional running
    '''

    def __init__(self, wid):
        self.wid = wid                          #工号
        self.env = gym.make(GAME).unwrapped     #创建环境
        self.env.seed(wid * 100 + RANDOMSEED)   #设置不同的随机种子，因为不希望每个worker的都一致
        self.ppo = GLOBAL_PPO                   #算法

    def work(self):
        '''
        Define a worker
        :return: None
        '''
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():                  #从COORD接受消息，看看是否应该should_stop
            s = self.env.reset()                    
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []   #记录data
            t0 = time.time()
            for t in range(EP_LEN):
                
                #看是否正在被更新。PPO进程正在工作，那么就在这里等待
                if not ROLLING_EVENT.is_set():  # 查询进程是否被阻塞，如果在阻塞状态，就证明如果global PPO正在更新。否则就可以继续。
                    ROLLING_EVENT.wait()        # worker进程的等待位置。wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                
                # 正常跑游戏，并搜集数据
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                #GLOBAL_UPDATE_COUNTER是每个work的在游戏中进行一步，也就是产生一条数据就会+1.
                #当GLOBAL_UPDATE_COUNTER大于batch(64)的时候，就可以进行更新。
                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE: # t == EP_LEN - 1 是最后一步
                    ## 计算每个状态对应的V(s')
                    ## 要注意，这里的len(buffer) < GLOBAL_UPDATE_COUNTER。所以数据是每个worker各自计算的。
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    ## 堆叠成数据，并保存到公共队列中。
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue

                    # 如果数据足够，就开始更新
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()   # stop collecting data
                        UPDATE_EVENT.set()      # global PPO update

                    if GLOBAL_EP >= EP_MAX:     # stop training
                        COORD.request_stop()    # 停止更新
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1

            print(
                'Episode: {}/{}  | Worker: {} | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    GLOBAL_EP, EP_MAX, self.wid, ep_r,
                    time.time() - t0
                )
            )


if __name__ == '__main__':

    # reproducible
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    GLOBAL_PPO = PPO()
    if args.train:  # train
        #定义两组不同的事件，update 和 rolling
        UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()  
        UPDATE_EVENT.clear()  # not update now，相当于把标志位设置为False
        ROLLING_EVENT.set()   # start to roll out，相当于把标志位设置为True，并通知所有处于等待阻塞状态的线程恢复运行状态。

        #创建workers
        workers = [Worker(wid=i) for i in range(N_WORKER)]

        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0     #全局更新次数计数器，全局EP计数器
        GLOBAL_RUNNING_R = []                       #记录动态的reward，看成绩
        COORD = tf.train.Coordinator()              #创建tensorflow的协调器
        QUEUE = queue.Queue()                       # workers putting data in this queue
        threads = []

        #为每个worker创建进程
        for worker in workers:  # worker threads
            t = threading.Thread(target=worker.work, args=())   #创建进程
            t.start()                                           #开始进程
            threads.append(t)                                   #把进程放到进程列表中，方便管理

        # add a PPO updating thread
        # 把一个全局的PPO更新放到进程列表最后。
        threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
        threads[-1].start()
        COORD.join(threads)                         #把进程列表交给协调器管理

        GLOBAL_PPO.save_ckpt()                      #保存全局参数

        # plot reward change and test
        plt.title('DPPO')
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.xlabel('Episode')
        plt.ylabel('Moving reward')
        plt.ylim(-2000, 0)
        plt.show()

    # test
    GLOBAL_PPO.load_ckpt()
    env = gym.make(GAME)
    while True:
        s = env.reset()
        for t in range(EP_LEN):
            env.render()
            s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
            if done:
                break
