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

# FrozenLake-v0是一个4*4的网络格子，每个格子可以是起始块，目标块、冻结块或者危险块。
# 我们的目标是让智能体学习如何从开始块如何行动到目标块上，而不是移动到危险块上。
# 智能体可以选择向上、向下、向左或者向右移动，同时游戏中还有可能吹来一阵风，将智能体吹到任意的方块上。
env = gym.make('FrozenLake-v0')

# 设置是否渲染，展示游戏画面。
render = False  
running_reward = None   

##================= Implement Q-Table learning algorithm =====================##

## 建立Q表格，并初始化为全0数组。形状为：[状态空间，动作空间]
Q = np.zeros([env.observation_space.n, env.action_space.n])

## 设置更新的超参数
## Q[s, a] = Q[s, a] + lr * (r + lambd * np.max(Q[s1, :]) - Q[s, a])
lr = .85                # Qleaning的学习率。alpha, if use value function approximation, we can ignore it
lambd = .99             # 折扣率 decay factor
num_episodes = 10000    # 迭代次数，也就是开始10000次游戏
rList = []              # 用于记录每次迭代的总奖励，这样我们就可以知道智能体是否有进步了。rewards for each episode


##=================开始游戏=====================##
for i in range(num_episodes):

    ## 重置环境初始状态
    episode_time = time.time()          #用于记录运行时间，我们可以通过比较运行时间判断算法效率。
    s, _ = env.reset()                  #重置初始状态。
    rAll = 0                            #用于记录这次游戏的总奖励，这里先初始化为0

    ## 开始Qlearning算法
    for j in range(99):
        if render: env.render()         #判断是否渲染环境。

        ## [敲黑板]
        ## 从Q表格中，找到当前状态S最大Q值，并在Q值上加上噪音。
        ## 然后找到最大的Q+噪音的动作
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

        ## 与环境互动，把动作放到env.step()函数，并返回下一状态S1，奖励，done，info
        s1, r, d, _ = env.step(a)

        ## 更新Q表格
        Q[s, a] = Q[s, a] + lr * (r + lambd * np.max(Q[s1, :]) - Q[s, a])
  
        rAll += r               # rAll累加当前的收获。
        s = s1                  # 把下一状态赋值给s，准备开始下一步。
        if d ==True:            # 如果已经到达最终状态，就跳出for循环。(开始下一次迭代)
            break

##=================更新结束，打印结果=====================##
    #每次的总收获都放到rlist。可以通过打印看看算法效率。
    rList.append(rAll)
    #每一次迭代获得的总收获rAll,会以0.01的份额加入到running_reward。(原代码这里rAll用了r，个人认为是rAll更合适)
    running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
    print("Episode [%d/%d] sum reward: %f running reward: %f took: %.5fs " % \
        (i, num_episodes, rAll, running_reward, time.time() - episode_time))

#最后打印Q表格，看看Q表格的样子吧。
print("Final Q-Table Values:/n %s" % Q)
