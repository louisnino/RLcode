import random
import numpy as np
import copy
from player import MCTSPlayer, Player, Human_Player, MCTSPlayer, Pure_MCTS_Player

def one_hot(x):
    arr = np.zeros(12)
    if x != -1:
        arr[x] = 1
    return arr

def one_hot_to_fig(arr):
    for i in range(arr.shape[0]):
        if arr[i] == 1:
            break
    return i

def count_one(arr,x):
    z = 0
    for a in arr:
        if a == x:
            z+=1
    return z
            
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def get_avail_act(arr):
    avail_act_list = []
    for i in range(len(arr)):
        if arr[i]==1:
            avail_act_list.append(i)
    return avail_act_list

class Blackjack():
    def __init__(self,game_state,policy_value_net):
        self.state = game_state
        self.figures = self.init_figures()
        self.buffer_value = []
        self.policy_value_net = policy_value_net
        self.p1 = MCTSPlayer(self.state,'p1',self.policy_value_net.policy_value_fn,n_playout=100,is_selfplay=1) #用于训练是selfplay
        self.p2 = MCTSPlayer(self.state,'p2',self.policy_value_net.policy_value_fn,n_playout=1000,is_selfplay=0) #用于真正自己玩
        self.human = Human_Player('human')
        self.random_player = Player('random')
        self.pure_tree_playre = Pure_MCTS_Player(self.state,'pure_tree',self.policy_value_net.policy_value_fn,n_playout=1000,is_selfplay=0)

    # 初始化数字池和可用数字
    def init_figures(self):
        figures = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for _ in range(2):
            figures.append(random.randint(1, 10))
        self.figures = np.array(figures)
        self.availabel_figures = np.ones(12)
        self.state.update_current_state(self.figures,self.availabel_figures,p1_num=False,p2_num=False,p1_choi=-1,p2_choi=-1)
        
    # 初始化玩家的数字
    def init_player_figures(self):
        a = random.randint(15, 21)
        b = random.randint(19, 27)
        # 两者相加不能大于21*2
        if a + b >= 21 * 2:
            a = a - (a + b - 21 * 2) / 2
            b = b - (a + b - 21 * 2) / 2
            a = int(a) - 1
            b = int(b) - 1
        # 两者相加要为奇数
        if (a + b) % 2 == 0:
            b = b - 1
        # 随机P1和P2
        if random.random() >= .5:
            self.p1_num = a
            self.p2_num = b
        else:
            self.p1_num = b
            self.p2_num = a
        self.state.current_state[2]=self.p1_num
        self.state.current_state[3]=self.p2_num
        return self.p1_num, self.p2_num   
    
        
    def who_first(self):
        if self.p1.num >= self.p2.num:
            return self.p1,self.p2
        else:
            return self.p2,self.p1
            

    def get_winner(self):
        if count_one(self.state.current_state[1],1)<=2:
            if self.state.current_state[2][0]<=21 and self.state.current_state[3][0]<=21:
                if self.state.current_state[2][0]>=self.state.current_state[3][0]:
                    winner = 0
                else:
                    winner = 1
            elif self.state.current_state[3][0]>21:
                winner = 0
            else:
                winner = 1
        return winner
            
    # 开始游戏(和纯树玩家玩)
    def start_game(self):        
        print('=========START GAME==========')
        self.init_figures()                             # 初始化数字池
        self.init_player_figures()                      # 初始化双方数字
        self.state.save_current_state()                 
        
        #print(self.state.current_state)
        for i in range(5):
            #=====打印状态=====
            print('********ROUND  %i*********'%(i+1))
            print(self.state.current_state[0])
            print(self.state.current_state[1])
            print(self.state.current_state[2])
            print(self.state.current_state[3])

            if self.state.current_state[2][0]>self.state.current_state[3][0]:           #如果第3行大于第4行，就纯树先走。
                act, num = self.pure_tree_playre.get_action(self.state.current_state)   # 纯树玩家选择
                #act, num = self.random_player.get_action(self.state.current_state)     # 随机玩家选择
                self.state.do_move(act)
                print('PTreePlayer Selcet No.%i  fig: %i '%(act,self.state.current_state[0][act-1]))
                
                act_2nd, num_2nd = self.p2.get_action(self.state.current_state)         # MCTS玩家选择
                self.state.do_move(act_2nd)
                print('MCTSPlayer  Selcet No.%i  fig: %i '%(act_2nd,self.state.current_state[0][act_2nd-1]))
            else:
                act_2nd, num_2nd = self.p2.get_action(self.state.current_state)         # MCTS玩家选择
                self.state.do_move(act_2nd)
                print('MCTSPlayer  Selcet No.%i  fig: %i '%(act_2nd,self.state.current_state[0][act_2nd-1]))
                act, num = self.pure_tree_playre.get_action(self.state.current_state)   # 纯树玩家选择
                #act, num = self.random_player.get_action(self.state.current_state)
                self.state.do_move(act)
                print('PTreePlayer Selcet No.%i  fig: %i '%(act,self.state.current_state[0][act-1]))
                
        if count_one(self.state.current_state[1],1)<=2:                                 # 判断是否已经结束
            if self.state.current_state[2][0]<=21 and self.state.current_state[3][0]<=21:   #如果两者都小于21，那么大的一方获胜
                if self.state.current_state[2][0]>=self.state.current_state[3][0]:
                    winner = 0              # 纯树
                else:
                    winner = 1              # MCTS
            elif self.state.current_state[3][0]>21:
                winner = 0                  # 纯树
            else:
                winner = 1                  # 纯树
        
        return winner

    # 开始游戏(和真人玩家玩)
    def start_game_human(self):        
        print('=========START GAME==========')
        self.init_figures()
        self.init_player_figures()
        self.state.save_current_state()
        
        #print(self.state.current_state)
        for i in range(5):
            print('********ROUND  %i*********'%(i+1))
            num_list = []
            for i in range(12):
                if self.state.current_state[1][i]==1:
                    num_list.append(int(self.state.current_state[0][i]))
                else:
                    num_list.append(0)
            print('数字列表:   : ', num_list)
            print('行动列表:   : ', list(range(1,13)))
            print('your number: ', self.state.current_state[2][0])
            print('oppe number: ', self.state.current_state[3][0])
            if self.state.current_state[2][0]>self.state.current_state[3][0]: #p1先手，p1是random 或者是玩家
                act, num = self.human.get_action(self.state.current_state)
                self.state.do_move(act)
                print('你的选择：[%i]  数字: [%i] '%(act+1,self.state.current_state[0][act]))
                
                act_2nd, num_2nd = self.p2.get_action(self.state.current_state)
                self.state.do_move(act_2nd)
                print('对手选择：[%i]  数字: [%i] '%(act_2nd+1,self.state.current_state[0][act_2nd]))
            else:
                act_2nd, num_2nd = self.p2.get_action(self.state.current_state)
                self.state.do_move(act_2nd)
                print('对手选择：[%i]  数字: [%i] '%(act_2nd+1,self.state.current_state[0][act_2nd]))
                act, num = self.human.get_action(self.state.current_state)
                self.state.do_move(act)
                print('你的选择：[%i]  数字: [%i] '%(act+1,self.state.current_state[0][act]))
                
        if count_one(self.state.current_state[1],1)<=2:
            if self.state.current_state[2][0]<=21 and self.state.current_state[3][0]<=21:
                if self.state.current_state[2][0]>=self.state.current_state[3][0]:
                    winner = 0
                else:
                    winner = 1
            elif self.state.current_state[3][0]>21:
                winner = 0
            else:
                winner = 1
        
        return winner
      
    def start_self_play(self): 
        
        states, mcts_probs, current_players, buffer_value = [], [], [],[]
        run_down_list = []
   
        self.init_figures()                     # 初始化数字公共数字
        self.init_player_figures()              # 初始化玩家自己的数字
        self.state.save_current_state()         # 保存到current_state
        
        #=====start a selfplay game=======
        for _ in range(5):                      # 进行5轮游戏
            #通过state判断谁先手。
            if self.state.current_state[2][0] > self.state.current_state[3][0]: 
                #run_down_list主要记录哪个player先手
                run_down_list.append(0)
                run_down_list.append(1)
            else:
                run_down_list.append(1)
                run_down_list.append(0) 

            #【敲黑板】选择1个动作。这个动作的选择，是根据MCTS模拟获得的。
            act1, act1_porbs = self.p1.get_action(self.state.current_state)
            self.state.do_move(act1)                    # 执行动作，并进入下一个state
            states.append((copy.copy(self.state.current_state)).reshape(-1,6,12,1).astype('float32'))   #加入到states保存，等会拿来训练网络
            mcts_probs.append(np.array(act1_porbs).astype('float32'))                                   #把act1_porbs保存，等会拿来训练网络
            
            #print('======change player========')
            act2, act2_porbs = self.p1.get_action(self.state.current_state)
            self.state.do_move(act2)
            states.append((copy.copy(self.state.current_state)).reshape(-1,6,12,1).astype('float32'))
            mcts_probs.append(np.array(act2_porbs).astype('float32'))                                   

        # 经过5轮之后，计算winner           
        winner = self.get_winner()
        if winner == 0:
            print('winner: p1')
        else:
            print('winner: p2')

        # 根据胜负，放入到最后
        for p in run_down_list:
            if p != winner:
            #if p == winner:
                buffer_value.append(np.ones(12).astype('float32'))
            else:
                buffer_value.append((np.ones(12) * (-1)).astype('float32'))

        self.p1.reset_player()
        self.p2.reset_player() 
        
        #把state，动作概率，结果返回。
        return zip(states, mcts_probs, buffer_value)

class Game_State():
    def __init__(self):
        self.current_state = np.zeros((6, 12))
        self.state_buffer = []
        
    #save current state
    def save_current_state(self):
        self.state_buffer.append(np.copy(self.current_state))
        #print(self.current_state)       
    
    #ini current state
    def update_current_state(self,figures,availabel_figures,p1_num,p2_num,p1_choi,p2_choi):
        self.current_state[0] = figures                 #可选数字
        self.current_state[1] = availabel_figures       #
        self.current_state[2] = p1_num * np.ones(12)
        self.current_state[3] = p2_num * np.ones(12)
        self.current_state[4] = one_hot(-1)
        self.current_state[5] = one_hot(-1) 
        
    def do_move(self,act):
        #=======先手=======
        if count_one(self.current_state[1],1)%2 == 0: 
            self.current_state[1][act] = 0
            #比较两个num谁大，看看谁先选
            if self.current_state[2][0]>self.current_state[3][0]:   #p1先手
                self.current_state[4] = one_hot(act)
            else:
                self.current_state[5] = one_hot(act)
            #self.save_current_state()
                
        #=======后手=======
        else:
            self.current_state[1][act] = 0
            if self.current_state[2][0]>self.current_state[3][0]:  #p1先手
                #print('++++++++act  :',act)
                #print('++++++++state:',self.current_state)
                self.current_state[5] = one_hot(act)
            else:
                self.current_state[4] = one_hot(act)
            self.cal_state()
            self.save_current_state()
        
    def cal_state(self):
        p1_choise = self.current_state[0][np.argwhere(self.current_state[4]==1)[0]]
        p2_choise = self.current_state[0][np.argwhere(self.current_state[5]==1)[0]]
        p1_num = self.current_state[2][0]
        p2_num = self.current_state[3][0]
        ab = abs(p1_choise-p2_choise)
        if p1_choise>=p2_choise:
            p1_num += ab
            p2_num -= ab
        else:
            p1_num -= ab
            p2_num += ab
        self.current_state[2] = p1_num * np.ones(12)
        self.current_state[3] = p2_num * np.ones(12)
        self.current_state[4] = np.zeros(12)
        self.current_state[5] = np.zeros(12)
    
    def get_curr_player(self):
        #看谁的数字大，大的先
        if count_one(self.current_state[4],1)>=1:
            return 1
        elif count_one(self.current_state[5],1)>=1:
            return 0
        else:
            if self.current_state[2][0]>self.current_state[3][0]:            
                return 0
            else:
                return 1