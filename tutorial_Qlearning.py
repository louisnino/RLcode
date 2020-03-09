"""Q-Table learning algorithm.

Non deep learning - TD Learning, Off-Policy, e-Greedy Exploration

Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))

See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.

For Q-Network, see tutorial_frozenlake_q_network.py

EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327

tensorflow==2.0.00
tensorlayer==2.0.0

"""

import time

import gym
import numpy as np

## Load the environment
env = gym.make('FrozenLake-v0')
#FrozenLake-v0是一个4*4的网络格子，每个格子可以是起始块，目标块、冻结块或者危险块。
# 我们的目标是让agent学习从开始块如何行动到目标块上，而不是移动到危险块上。
# agent可以选择向上、向下、向左或者向右移动，同时游戏中还有可能吹来一阵风，将agent吹到任意的方块上。
render = False  # display the game environment
running_reward = None

##================= Implement Q-Table learning algorithm =====================##
## Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
## Set learning parameters
lr = .85  # alpha, if use value function approximation, we can ignore it
lambd = .99  # decay factor
num_episodes = 10000
rList = []  # rewards for each episode
for i in range(num_episodes):
    ## Reset environment and get first new observation
    episode_time = time.time()
    s = env.reset()
    rAll = 0
    ## The Q-Table learning algorithm
    for j in range(99):
        if render: env.render()
        ## Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        ## Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        ## Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + lambd * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if d ==True:
            break
    rList.append(rAll)
    running_reward = r if running_reward is None else running_reward * 0.99 + r * 0.01
    print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % \
        (i, num_episodes, rAll, running_reward, time.time() - episode_time))

print("Final Q-Table Values:/n %s" % Q)
