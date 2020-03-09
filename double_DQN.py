import tensorflow as tf
import tensorlayer as tl 
from collections import deque
import numpy as np 
import gym
import random


class Double_DQN():
    def __init__(self):

        self.env = gym.make('CartPole-v0')
        self.input_dim = self.env.observation_space.shape[0]
        self.Q_network = self.get_model()
        self.Q_network.train()
        self.target_Q_network = self.get_model()
        self.target_Q_network.eval()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=2000) #队列，最大值是2000
        self.batch = 128
        self.gamma = 0.95 #折扣率
        self.learning_rate = 1e-3 #学习率
        self.opt = tf.optimizers.Adam(self.learning_rate)  #优化器
        self.is_rend = False  #默认不渲染，当达到一定次数后，开始渲染。

    def get_model(self):
        self.input = tl.layers.Input(shape=[None,self.input_dim])
        self.h1 = tl.layers.Dense(32, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(self.input)
        self.h2 = tl.layers.Dense(16, tf.nn.relu, W_init=tf.initializers.GlorotUniform())(self.h1)
        self.output = tl.layers.Dense(2,act=None, W_init=tf.initializers.GlorotUniform())(self.h2)
        return tl.models.Model(inputs=self.input,outputs=self.output)

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_Q(self):
        for i , target in zip(self.Q_network.trainable_weights, self.target_Q_network.trainable_weights):
            target.assign(i)
        #print('updata success!')

    def remember(self, s, a, s_, r, done):
        data = (s, a, s_, r, done)
        self.memory.append(data)

    def process_data(self):
        data = random.sample(self.memory, self.batch)
        s = np.array([d[0] for d in data])
        a = [d[1] for d in data]
        s_ = np.array([d[2] for d in data])
        r = [d[3] for d in data]
        done = [d[4] for d in data]

        # DQN的target
        '''
        target_Q = np.max(self.target_Q_network(np.array(s_,dtype='float32')))  #计算下一状态最大的Q值
        target = target_Q * self.gamma + r
        '''
        #double的target
        y = self.Q_network(np.array(s,dtype='float32'))
        y = y.numpy()
        Q1 = self.target_Q_network(np.array(s_,dtype='float32'))
        Q2 = self.Q_network(np.array(s_,dtype='float32'))
        next_action = np.argmax(Q2,axis=1)
        
        for i ,(_,a,_,r,done) in enumerate(data):
            if done:
                target = r
            else:
                target = r + self.gamma * Q1[i][next_action[i]]
            #print('====y.numpy()[i]:',y[i])  
            #print('====y.numpy()[i][a]:',y[i][a])
            #print('====target:',target)
            target = np.array(target,dtype='float32')
            
            #print('====target:',target)
            y[i][a] = target
            #print('====y.numpy()[i]:',y[i])
            #print('===========')

            

        return s, y

    def update_Q_network(self):
        s,y = self.process_data()
        with tf.GradientTape() as tape:
            Q = self.Q_network(np.array(s,dtype='float32'))
            #loss = tf.losses.BinaryCrossentropy(y,Q)
            #loss = tf.reduce_mean(tf.square(Q, y))
            loss = tl.cost.mean_squared_error(Q,y)
        #print('====y:',y)
        #print('====Q:',Q)
        grads = tape.gradient(loss, self.Q_network.trainable_weights)
        self.opt.apply_gradients(zip(grads,self.Q_network.trainable_weights))
        return loss

    def get_action(self,s):  #epsilon-greedy
        if np.random.rand()>=self.epsilon:
            q = self.Q_network(np.array(s,dtype='float32').reshape([-1,4]))    
            a = np.argmax(q)
            #print('greedy:',a) 
            return a
        else:
            a = random.randint(0, 1)
            #print('random:',a) 
            return a
            



    def train(self,episode):
        step = 0
        rend = 0
        for ep in range(episode):

            s = self.env.reset()
            total_reward = 0
            total_loss = []
            loss = 0 
            while True:
                if self.is_rend:self.env.render()
                a = self.get_action(s)
                s_,r,done,_ = self.env.step(a)

                total_reward += r
                step += 1

                #保存s, a, s_, r, done
                self.remember(s, a, s_, r, done)
                s = s_

                if len(self.memory)>self.batch:
                    loss = self.update_Q_network()
                    total_loss.append(loss)
                    if (step+1)%5 == 0:
                        self.update_epsilon()
                        self.update_target_Q()

                if done:
                    print('EP:%i,  total_rewards:%f,   epsilon:%f, loss:%f'%(ep,total_reward,self.epsilon,np.mean(loss)))
                    break

            if total_reward>=200:
                rend += 1
                if rend == 5:
                    #print('start rend')
                    self.is_rend = True
                    

if __name__=='__main__':
    ddqn = Double_DQN()
    ddqn.train(200)
