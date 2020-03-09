"""
Deep Q-Network Q(a, s)
-----------------------
TD Learning, Off-Policy, e-Greedy Exploration (GLIE).

Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))
delta_w = R + lambda * Q(newS, newA)

See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.

Reference
----------
original paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327

Note: Policy Network has been proved to be better than Q-Learning, see tutorial_atari_pong.py

Environment
-----------
# The FrozenLake v0 environment
https://gym.openai.com/envs/FrozenLake-v0
The agent controls the movement of a character in a grid world. Some tiles of
the grid are walkable, and others lead to the agent falling into the water.
Additionally, the movement direction of the agent is uncertain and only partially
depends on the chosen direction. The agent is rewarded for finding a walkable
path to a goal tile.
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
The episode ends when you reach the goal or fall in a hole. You receive a reward
of 1 if you reach the goal, and zero otherwise.

Prerequisites
--------------
tensorflow>=2.0.0a0
tensorlayer>=2.0.0

To run
-------
python tutorial_DQN.py --train/test


"""
import argparse
import time

import gym
import numpy as np
import tensorflow as tf

import tensorlayer as tl

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

tl.logging.set_verbosity(tl.logging.DEBUG)

#####################  hyper parameters  ####################
lambd = .99  # decay factor
e = 0.1  # e-Greedy Exploration, the larger the more random
num_episodes = 10000
render = False  # display the game environment
running_reward = None

##################### DQN ##########################


def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.

def get_model(inputs_shape):
    '''
    定义模型
    注意输入的shape和输出的shape
    '''
    ni = tl.layers.Input(inputs_shape, name='observation')
    nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")


def save_ckpt(model):  # save trained weights
    '''
    保存参数
    '''
    tl.files.save_npz(model.trainable_weights, name='dqn_model.npz')


def load_ckpt(model):  # load trained weights
    '''
    加载参数
    '''
    tl.files.load_and_assign_npz(name='dqn_model.npz', network=model)


if __name__ == '__main__':

    qnetwork = get_model([None, 16]) #定义inputshape[None,16]。16是state数量
    qnetwork.train()  #调用tl的时候，需要标注这个模型是否可以调用。
    train_weights = qnetwork.trainable_weights #模型的参数

    optimizer = tf.optimizers.SGD(learning_rate=0.1)   #定义优化器
    env = gym.make('FrozenLake-v0') #定义环境

    if args.train:
        t0 = time.time()
        for i in range(num_episodes):
            ## Reset environment and get first new observation
            # episode_time = time.time()
            s = env.reset()  # observation is state, integer 0 ~ 15
            rAll = 0
            for j in range(99):  # step index, maximum step is 99
                if render: env.render()
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                ## 把state放入network，计算Q值。
                ## 注意，这里先把state进行onehote处理，这里注意解释下什么是onehot
                ## 输出，这个状态下，所有动作的Q值，也就是说，是一个[None,4]大小的矩阵
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                # 在矩阵中找最大的Q值的动作
                a = np.argmax(allQ, 1)

                ## e-Greedy Exploration !!! sample random action
                # e-Greedy：如果小于epsilon，就随机。否则就不用随机了
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                ## Get new state and reward from environment
                # 输入到环境，获得下一步的state，reward，done
                s1, r, d, _ = env.step(a[0])
                ## Obtain the Q' values by feeding the new state through our network
                # 把new-state 放入，预测下一个state的Q值。
                Q1 = qnetwork(np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()

                
                ##=======计算target=======
                ## Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
                targetQ = allQ
                targetQ[0, a[0]] = r + lambd * maxQ1
                ## Train network using target and predicted Q values
                # it is not real target Q value, it is just an estimation,
                # but check the Q-Learning update formula:
                #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)

                ## 利用自动求导 进行更新。
                with tf.GradientTape() as tape:
                    _qvalues = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32))
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                # 累计reward，并且把s更新为newstate
                rAll += r
                s = s1
                ## Reduce chance of random action if an episode is done.

                #update-epsilond
                if d ==True:
                    e = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
                    break

            ## Note that, the rewards here with random action
            ## 这里的running_reward用于记载每一次更新的总和。为了能够更加看清变化，所以大部分是前面的。只有一部分是后面的。
            running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
            # print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % \
            #     (i, num_episodes, rAll, running_reward, time.time() - episode_time))
            print('Episode: {}/{}  | Episode Reward: {:.4f} | Running Average Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(i, num_episodes, rAll, running_reward,  time.time()-t0 ))
        save_ckpt(qnetwork)  # save model
    

    ##============这部分是正式游戏了======
    if args.test:
        t0 = time.time()
        load_ckpt(qnetwork)  # load model
        for i in range(num_episodes):
            ## Reset environment and get first new observation
            episode_time = time.time()
            s = env.reset()  # observation is state, integer 0 ~ 15
            rAll = 0
            for j in range(99):  # step index, maximum step is 99
                if render: env.render()
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)  # no epsilon, only greedy for testing

                ## Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                rAll += r
                s = s1
                ## Reduce chance of random action if an episode is done.
                if d ==True:
                    e = 1. / ((i / 50) + 10)  # reduce e, GLIE: Greey in the limit with infinite Exploration
                    break

            ## Note that, the rewards here with random action
            running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
            # print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % \
            #     (i, num_episodes, rAll, running_reward, time.time() - episode_time))
            print('Episode: {}/{}  | Episode Reward: {:.4f} | Running Average Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(i, num_episodes, rAll, running_reward,  time.time()-t0 ))
