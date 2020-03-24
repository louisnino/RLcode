import tensorflow as tf
import tensorlayer as tl
import numpy as np
#from game import Game
import os

class PolicyValueNet():
    def __init__(self):
        self.model = self.get_model()
        self.model.train()
        self.old_model = self.get_model()
        self.old_model.eval()
        self.learning_rate = 1e-5
        self.opt = tf.optimizers.Adam(self.learning_rate)
    
    def get_model(self):      
        #=====init W=====
        w_init = tf.random_normal_initializer(stddev=0.1)
        #=====Input=====
        inn = tl.layers.Input([None,4,4,17])
        #transpose = tf.transpose(inn,[0,2,3,1])
        #=====Conv======        
        conv1 = tl.layers.Conv2d(n_filter=32,filter_size=(4,4),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(inn)
        conv2 = tl.layers.Conv2d(n_filter=64,filter_size=(3,3),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(conv1)
        conv3 = tl.layers.Conv2d(n_filter=128,filter_size=(2,2),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(conv2)
        #====Action Network=====
        action_conv = tl.layers.Conv2d(n_filter=4,filter_size=(1,1),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(conv3)
        action_reshape = tl.layers.Reshape([-1,4*16])(action_conv)
        action_fc = tl.layers.Dense(n_units=16*5,act=tf.nn.log_softmax,b_init=None)(action_reshape)
        action_output = tl.layers.Reshape([-1,16,5])(action_fc)
        #=====Value Network=====
        value_conv = tl.layers.Conv2d(n_filter=2,filter_size=(1,1),act=tl.activation.leaky_relu,W_init=w_init,b_init=None,padding='SAME')(conv3)
        value_reshape = tl.layers.Reshape([-1,2*16])(value_conv)
        value_fc1 = tl.layers.Dense(n_units=2*16,act=tl.activation.leaky_relu,W_init=w_init,b_init=None,)(value_reshape)
        value_fc2 = tl.layers.Dense(n_units=1,act=tf.nn.tanh)(value_fc1)
        
        return tl.models.Model(inputs=inn, outputs=[action_output,value_fc2])
    
    def policy_value(self,state):
        '''
        通过model计算动作概率，和value值
        '''
        state = np.reshape(state,[-1,4,4,17]).astype('float32')
        log_act_porbs, value = self.model(state)
        act_porbs = np.exp(log_act_porbs)
        return act_porbs, value 

    def policy_value_fn(self,game_board):
        #game_board先转为state才能放入运算,但怎么需要player呢？
        legal_action = game_board.get_legal_action(game_board.board)

        state = game_board.board_to_state(game_board.board)
        act_porbs, value = self.policy_value(np.array(state).astype('float32'))
        act_porbs = legal_action * act_porbs  #这是一个带有概率的数组
        return act_porbs, value

    def policy_value_old(self,state):
        state = np.reshape(state,[-1,4,4,17]).astype('float32')
        log_act_porbs, value = self.model(state)
        act_porbs = np.exp(log_act_porbs)
        return act_porbs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        with tf.GradientTape() as tape:
            act_porbs, value = self.policy_value(state_batch)

            #policy 和 value loss
            self.value_loss = tf.losses.mean_squared_error(winner_batch, value)
            print('mcts_probs',mcts_probs)
            print('act_porbs',act_porbs)
            self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(mcts_probs,act_porbs),1)))

            #L2 penalty
            l2_penalty_beta = 1e-4
            var = self.model.trainable_weights
            l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in var if 'bias' not in v.name.lower()])

            #total loss
            self.loss = self.value_loss + self.policy_loss + l2_penalty

        grads = tape.gradient(self.loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))

        total_loss = tf.reduce_mean(self.loss)
        value_loss = tf.reduce_mean(self.value_loss)
        policy_loss = tf.reduce_mean(self.policy_loss)
        l2_penalty = tf.reduce_mean(l2_penalty)

        return total_loss,value_loss,policy_loss,l2_penalty
        
    def update_parm(self):
        '''
        更新旧模型参数
        '''
        for i, old in zip(self.model.trainable_weights, self.old_model.trainable_weights):
            old.assign(i)


    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/blackjack.hdf5', self.model)


    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/blackjack.hdf5', self.model)   

