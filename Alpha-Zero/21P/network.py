import tensorflow as tf
import tensorlayer as tl
import os
import numpy as np

def get_avail_act(arr):
    avail_act_list = []
    for i in range(len(arr)):
        if arr[i]==1:
            avail_act_list.append(i)
    return avail_act_list


class PolicyValueNet():
    def __init__(self):
        self.lr =1e-5
        self.opt = tf.optimizers.Adam(self.lr)
        
        self.model = self.get_model()
        self.model.train()
        self.old_model = self.get_model()
        self.old_model.eval()
        self.model_save_path = './model/blackjack.hdf5'
        
    def update_param(self):
        '''
        赋值给oldmodel
        '''
        for i, old in zip(self.model.trainable_weights, self.old_model.trainable_weights):
            old.assign(i)
        
    def get_model(self):
        
        #=====init W=====
        w_init = tf.random_normal_initializer(stddev=0.1)
        
        #=====Input=====
        inn = tl.layers.Input([None,6,12,1])

        #=====Conv======        
        conv1 = tl.layers.Conv2d(n_filter=32,filter_size=(6,3),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(inn)
        conv2 = tl.layers.Conv2d(n_filter=64,filter_size=(6,3),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(conv1)
        conv3 = tl.layers.Conv2d(n_filter=128,filter_size=(6,3),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(conv2)
        #====Action Network=====
        action_conv = tl.layers.Conv2d(n_filter=4,filter_size=(6,1),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(conv3)
        action_reshape = tl.layers.Reshape([-1,4*6*12])(action_conv)
        action_fc = tl.layers.Dense(n_units=12,act=tf.nn.log_softmax,b_init=None,)(action_reshape)
        #=====Value Network=====
        value_conv = tl.layers.Conv2d(n_filter=2,filter_size=(6,1),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(conv3)
        value_reshape = tl.layers.Reshape([-1,2*6*12])(value_conv)
        value_fc1 = tl.layers.Dense(n_units=2*12,act=tl.activation.leaky_relu,W_init=w_init,b_init=None,)(value_reshape)
        value_fc2 = tl.layers.Dense(n_units=1,act=tf.nn.tanh)(value_fc1)
        
        return tl.models.Model(inputs=inn, outputs=[action_fc,value_fc2])
    
    def policy_value_fn(self, state):
        legal_positions = get_avail_act(state.current_state[1])
        #print('legal_positions',legal_positions)
        current_state = state
        act_probs, value ,_= self.policy_value(
            state.current_state.reshape(-1, 6,12,1).astype('float32')
            )

        #print("++++++++++expand+++++++++++")
        #print("legal_positions",legal_positions)
        #print("act_probs      ",act_probs.flatten()[legal_positions])
        #print("+++++++++++++++++++++++++++")

        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])

        return act_probs, value[0][0]
    
    def policy_value(self,state):
        log_act_probs,value = self.model(state)
        act_probs = np.exp(log_act_probs)
        return act_probs, value , log_act_probs
    
    def policy_value_old(self,state):
        log_act_probs,value = self.old_model(state)
        act_probs = np.exp(log_act_probs)
        return act_probs, value
    
    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        '''
            开始训练
        '''
        with tf.GradientTape() as tape:           
            act_probs, value , log_act_probs= self.policy_value(state_batch)
            
            # value_loss 用 mse 就可以了
            self.value_loss = tf.losses.mean_squared_error(winner_batch,value)
            self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(mcts_probs, log_act_probs), 1)))
            
            #print('=====================================')
            #print('mcts_probs')
            #print(mcts_probs[0])
            #print('log_act_probs')
            #print(log_act_probs[0])
            #print('=====================================')
            
            #L2权重正则化,防止过拟合
            l2_penalty_beta = 1e-4
            vars = self.model.trainable_weights
            l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
            
            self.loss = self.value_loss + self.policy_loss + l2_penalty
        
        #自动求导，常规动作
        grads = tape.gradient(self.loss,self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads,self.model.trainable_weights))
        loss = tf.reduce_mean(self.loss)
        value_loss = tf.reduce_mean(self.value_loss)
        policy_loss = tf.reduce_mean(self.policy_loss)
        l2_penalty = tf.reduce_mean(l2_penalty)
        
        return loss,value_loss,policy_loss,l2_penalty
    
    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5(self.model_save_path, self.model)


    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order(self.model_save_path, self.model)   