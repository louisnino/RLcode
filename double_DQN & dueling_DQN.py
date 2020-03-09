import tensorflow as tf
import tensorlayer as tl 
from collections import deque
import numpy as np 
import gym
import random


class Double_DQN():
    def __init__(self):
        self.env = gym.make('CartPole-v0')                      #定义环境
        self.input_dim = self.env.observation_space.shape[0]    #定义网络的输入形状，这里就是输入S

        #建立两个网络
        self.Q_network = self.get_model()                       #建立一个Q网络
        self.Q_network.train()                                  #在tensorlayer要指定这个网络用于训练。
        self.target_Q_network = self.get_model()                #创建一个target_Q网络
        self.target_Q_network.eval()                            #这个网络指定为不用于更新。

        ## epsilon-greedy相关参数
        self.epsilon = 1.0                  #epsilon大小，随机数大于epsilon，则进行开发；否则，进行探索。   
        self.epsilon_decay = 0.995          #减少率：epsilon会随着迭代而更新，每次会乘以0.995
        self.epsilon_min = 0.01             #小于最小epsilon就不再减少了。

        #其余超参数
        self.memory = deque(maxlen=2000)    #队列，最大值是2000
        self.batch = 128
        self.gamma = 0.95                   #折扣率
        self.learning_rate = 1e-3           #学习率
        self.opt = tf.optimizers.Adam(self.learning_rate)  #优化器
        self.is_rend = False                #默认不渲染，当达到一定次数后，开始渲染。
    '''
    def get_model(self):
        #创建网络
        #    输入：S
        #    输出：所有动作的Q值
        self.input = tl.layers.Input(shape=[None,self.input_dim])
        self.h1 = tl.layers.Dense(32, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(self.input)
        self.h2 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(self.h1)
        self.output = tl.layers.Dense(2,act=None, W_init=tf.initializers.GlorotUniform())(self.h2)
        return tl.models.Model(inputs=self.input,outputs=self.output)

    '''
    
    # dueling DQN只改了网络架构。
    def get_model(self):
        #第一部分
        input = tl.layers.Input(shape=[None,self.input_dim])
        h1 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(input)
        h2 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(h1)
        #第二部分
        svalue = tl.layers.Dense(2,)(h2)
        #第三部分
        avalue = tl.layers.Dense(2,)(h2)                                                    #计算avalue
        mean = tl.layers.Lambda(lambda x: tf.reduce_mean(x,axis=1,keepdims=True))(avalue)   #用Lambda层，计算avg(a)
        advantage = tl.layers.ElementwiseLambda(lambda x,y: x-y)([avalue,mean])             #a - avg(a)

        output = tl.layers.ElementwiseLambda(lambda x,y: x+y)([svalue,avalue])               
        return tl.models.Model(inputs=input,outputs=output)
    

    def update_epsilon(self):
        '''
        用于更新epsilon
            除非已经epsilon_min还小，否则比每次都乘以减少率epsilon_decay。
        '''
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_Q(self):
        '''
        Q网络学习完之后，需要把参数赋值到target_Q网络
        '''
        for i , target in zip(self.Q_network.trainable_weights, self.target_Q_network.trainable_weights):
            target.assign(i)

    def remember(self, s, a, s_, r, done):
        '''
        把数据放入到队列中保存。
        '''
        data = (s, a, s_, r, done)
        self.memory.append(data)

    def process_data(self):

        # 从队列中，随机取出一个batch大小的数据。
        data = random.sample(self.memory, self.batch)
        s = np.array([d[0] for d in data])
        a = [d[1] for d in data]
        s_ = np.array([d[2] for d in data])
        r = [d[3] for d in data]
        done = [d[4] for d in data]

        # 原始DQN的target
        '''
        target_Q = np.max(self.target_Q_network(np.array(s_,dtype='float32')))  #计算下一状态最大的Q值
        target = target_Q * self.gamma + r
        '''
        # [敲黑板]
        # 计算Double的target
        y = self.Q_network(np.array(s,dtype='float32'))
        y = y.numpy()
        Q1 = self.target_Q_network(np.array(s_,dtype='float32'))
        Q2 = self.Q_network(np.array(s_,dtype='float32'))
        next_action = np.argmax(Q2,axis=1)
        
        for i ,(_,a,_,r,done) in enumerate(data):
            if done:
                target = r
            else:
                #[敲黑板]
                # next_action是从Q_network计算出来的最大Q值的动作
                # 但输出的，是target_Q_network中的next_action的Q值。
                # 可以理解为：一个网络提议案，另外一个网络进行执行
                target = r + self.gamma * Q1[i][next_action[i]]
            target = np.array(target,dtype='float32')

            # y 就是更新目标。           
            y[i][a] = target
        return s, y

    def update_Q_network(self):
        '''
        更新Q_network，最小化target和Q的距离
        '''
        s,y = self.process_data()
        with tf.GradientTape() as tape:
            Q = self.Q_network(np.array(s,dtype='float32'))
            loss = tl.cost.mean_squared_error(Q,y)              # 最小化target和Q的距离
        grads = tape.gradient(loss, self.Q_network.trainable_weights)
        self.opt.apply_gradients(zip(grads,self.Q_network.trainable_weights))
        return loss

    def get_action(self,s):
        '''
        用epsilon-greedy的方式求动作。
        '''
        # 先随机一个数，如果比epsilon大，那么，就输出最大Q值的动作。
        if np.random.rand()>=self.epsilon:
            q = self.Q_network(np.array(s,dtype='float32').reshape([-1,4]))    
            a = np.argmax(q)
            return a
        # 否则，随机一个动作输出。
        else:
            a = random.randint(0, 1)
            return a

    ## 开始训练       
    def train(self,episode):
        step = 0
        rend = 0
        for ep in range(episode):

            s = self.env.reset()        #重置初始状态s
            total_reward = 0
            total_loss = []
            loss = 0 

            while True:
                if self.is_rend:self.env.render()

                # 进行游戏
                a = self.get_action(s)
                s_,r,done,_ = self.env.step(a)
                total_reward += r
                step += 1

                #保存s, a, s_, r, done
                self.remember(s, a, s_, r, done)
                s = s_

                #如果数据足够，那么就开始更新
                if len(self.memory)>self.batch:
                    loss = self.update_Q_network()
                    total_loss.append(loss)
                    if (step+1)%5 == 0:
                        self.update_epsilon()
                        self.update_target_Q()

                #如果到最终状态，就打印一下成绩如何
                if done:
                    print('EP:%i,  total_rewards:%f,   epsilon:%f, loss:%f'%(ep,total_reward,self.epsilon,np.mean(loss)))
                    break
            
            # 如果有5个ep成绩大于200，就开始渲染游戏。
            if total_reward>=200:
                rend += 1
                if rend == 5:
                    self.is_rend = True
                    
# 开始运行游戏
if __name__=='__main__':
    ddqn = Double_DQN()
    ddqn.train(200)
