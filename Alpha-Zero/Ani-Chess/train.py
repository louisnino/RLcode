import numpy as np
import random
from collections import defaultdict, deque
from game import Game, Board
from network import PolicyValueNet
import time

def list_mean(list):
    sum = 0.0
    for i in range(len(list)):
        sum += list[i]
    return sum/len(list)



class TrainPipeline():
    def __init__(self):
        self.policy_value_net = PolicyValueNet()
        self.game = Game(self.policy_value_net)
    
        self.game_batch_num = 10    #1000
        self.play_batch_size = 1
        self.batch_size = 64        #每次train用的数据块大小
        self.buffer_size = 512
        self.update_epochs = 16     #更新次数

        self.data_buffer = deque(maxlen=self.buffer_size)
        self.learning_rate = 2e-3
        self.lr_multiplier = 1
        self.kl_targ = 0.02

    def collect_selfplay_data(self, n_games=1):
        for _ in range(n_games):
            play_data = self.game.start_self_play()
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)

    def run(self):
        for i in range(self.game_batch_num):
            t0 = time.time()
            self.collect_selfplay_data()
            print("======batch :{}, episode_len:{}, time:{}======".format(i+1, self.episode_len, time.time()-t0))

            if len(self.data_buffer)>self.batch_size:
                loss = self.policy_update()
                self.policy_value_net.save_ckpt()

    def policy_update(self):

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        
        #print('minibatch_len:',len(mini_batch))
        state_batch = np.array([data[0] for data in mini_batch]).astype('float32')
        mcts_probs_batch = np.array([data[1] for data in mini_batch]).astype('float32')
        winner_batch = np.array([data[2] for data in mini_batch]).astype('float32')

        total_loss_list, value_loss_list, policy_loss_list, l2_penalty_list = [],[],[],[]
        
        '''
        for i in range(len(mini_batch)):
            state_batch = mini_batch[i][0]
            mcts_probs_batch = mini_batch[i][1]
            winner_batch = mini_batch[i][2]
        '''
        #更新old_network
        self.policy_value_net.update_parm()
        old_probs, old_v = self.policy_value_net.policy_value_old(state_batch)

        #print('state_batch:',state_batch)
        #print('mcts_probs_batch:',mcts_probs_batch)
        print('winner_batch:',winner_batch)

        for _ in range(self.update_epochs):
            loss= self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learning_rate*self.lr_multiplier)

            total_loss, value_loss, policy_loss, l2_penalty = loss

            total_loss_list.append(total_loss)
            value_loss_list.append(value_loss)
            policy_loss_list.append(policy_loss)
            l2_penalty_list.append(l2_penalty)

            print('total_loss: %f,value_loss: %f,policy_loss: %f,l2_penalty: %f'%(total_loss, value_loss, policy_loss, l2_penalty))

            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        # 根据上次更新的KL_diverges大小，动态调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        total_loss,
                        explained_var_old,
                        explained_var_new))
        total_loss_mean = list_mean(total_loss_list)
        value_loss_mean  = list_mean(value_loss_list)
        policy_loss_mean  = list_mean(policy_loss_list)
        l2_penalty_mean  = list_mean(l2_penalty_list)

        return total_loss_mean, value_loss_mean, policy_loss_mean, l2_penalty_mean              


if __name__ == "__main__":

    trainpipeline = TrainPipeline()
    trainpipeline.run()