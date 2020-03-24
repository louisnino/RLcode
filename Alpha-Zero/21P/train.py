from collections import defaultdict, deque
import ipywidgets as widgets	            # 控件库
from IPython.display import display	        # 显示控件的方法
import numpy as np
import random
from game import Game_State
from network import PolicyValueNet
from game import Blackjack

class TrainPipeline():

    def __init__(self,init_model=None):
        self.state = Game_State()
        self.policy_value_net = PolicyValueNet()
        self.game = Blackjack(self.state,self.policy_value_net)
        
        self.game_batch_num = 30            #相当于更新次数
        self.play_batch_size = 1            #跑多少次去获取batch
        self.batch_size = 8
        self.buffer_size = 512
        self.epochs = 32                    #更新多少次
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.learn_rate = 2e-3
        self.lr_multiplier = 1
        self.kl_targ = 0.02
        
    #好像最重要的是start_selfplay，其他保存的好像都可以的了。
    def collect_selfplay_data(self,n_games=1):
        '''
        收集selfplay的数据
        '''
        for _ in range(n_games):                        #n_games selfplay的次数
            play_data = self.game.start_self_play()     #开始selfplay，并返回数据给play_data
            play_data = list(play_data)[:]              
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)          #把selfplay加入大data_buffer

    def run(self,ep):
        #self.policy_value_net.load_ckpt()              #加载原来的参数继续训练
        for i in range(ep):                             #训练次数
            self.collect_selfplay_data(self.play_batch_size)

            print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
            if len(self.data_buffer)>self.batch_size:
                loss= self.policy_update()
                print('============================No%i update network SUCCESS==========================================='%(i))
            self.policy_value_net.save_ckpt()
                
    def policy_update(self):
        """update the policy-value net"""
        #========解压数据============
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]                  # state
        mcts_probs_batch = [data[1] for data in mini_batch]             # probs
        winner_batch = [data[2] for data in mini_batch]                 # winner
        
        #===========================
        #这里好像做了important sampling，直接计算KL_diverges大小，超过一定就早停
        self.policy_value_net.update_param()        
        old_probs, old_v = self.policy_value_net.policy_value_old(state_batch)

        #进行epochs次训练
        for _ in range(self.epochs):
            # 开始训练
            loss,value_loss,policy_loss,l2_penalty= self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            print('total_loss: %f , value_loss: %f , policy_loss: %f , l2_penalty: %f'%(loss,value_loss,policy_loss,l2_penalty))
            new_probs, new_v,_ = self.policy_value_net.policy_value(state_batch)
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
                        loss,
                        explained_var_old,
                        explained_var_new))
        return loss           

    def play_game(self,playtime=1):
        self.policy_value_net.load_ckpt()               #读取神经网络参数
        pure_tree_playre_win_count = 0
        tree_player_win_count = 0

        for _ in range(playtime):
            winner = self.game.start_game()
            if winner == 0:
                print('pure_tree_playre_win!!!')
                pure_tree_playre_win_count += 1
            else:
                print('MCTS_tree_player_win!!!')
                tree_player_win_count += 1

        print('===========Result============')
        print('pure_tree_playre_win: %i  '%(pure_tree_playre_win_count))
        print('MCTS_tree_player_win: %i  '%(tree_player_win_count))
        return tree_player_win_count

      
    def playgame_with_human(self,playtime=1):
        '''
        和真人玩
        '''
        self.policy_value_net.load_ckpt()           #加载参数
        human_win_count = 0
        msts_player_win_count = 0
        for i in range(playtime):
            winner = self.game.start_game_human()
            if winner == 0:
                print('Human_playre_win!!!')
                human_win_count += 1
            else:
                print('MCTS_tree_player_win!!!')                
                msts_player_win_count += 1
        print('===========Result============')
        print('Human_playre_win: %i  '%(human_win_count))
        print('MCTS_player_win: %i  '%(msts_player_win_count))

if __name__=="__main__":
    trainpipeline = TrainPipeline()
    #trainpipeline.run(5000)
    trainpipeline.play_game(10)

