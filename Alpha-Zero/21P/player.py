import random
import numpy as np
from mcts import MCTS

class Player():
    #def __init__(self, name, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
    def __init__(self, name,):
        self.name = name
        #self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        #self._is_selfplay = is_selfplay
        self.last_action = -1
        self.num = 0

    def get_action(self,state):
        while True:
            act = random.randint(0, 11)
            if state[1][act] == 1:
                break
        num = state[0][act]
        print('%s 选择第:%i个数字：%i'%(self.name,act+1,num))
        return act, num
    
class Human_Player():
    def __init__(self, name,):
        self.name = name
        self.last_action = -1
        self.num = 0

    def get_action(self,state):
        
        while True:
            act = int(input('choose action'))-1
            if act>12:
                print('your choise is over 12')
            if state[1][act] != 0:
                break
            else:
                print('your choise has been choosen')
        num = state[0][act]
        return act, num

class MCTSPlayer(object):
    def __init__(self,state,name,policy_value_function, c_puct=0.1, n_playout=500, is_selfplay=1):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.name = name
        self.last_action = -1
        self.num = 0
        self.state = state

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)                  #把-1传进去，就重置了整个树了。

    def get_action(self, state, temp=1e-3, return_prob=1):

        sensible_moves = np.argwhere(state[1]==1)       # 获得合法的选择
        move_probs = np.zeros(12)
        if len(sensible_moves) > 2:                     # 判断一下游戏是否应该结束了

            # 【敲黑板】进行n_playout模拟，生成一棵MCTS，返回根节点的acts, probs
            acts, probs = self.mcts.get_move_probs(self.state, temp, is_pure_tree=0)
            move_probs[list(acts)] = probs

            #=======================================
            #如果是selfplay模式，就要加0.25噪音。然后sample出一个move，执行。
            #如果不是selfplay模式，就不加噪音，但会重置整棵树。
            if self._is_selfplay: 
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )

                self.mcts.update_with_move(move)
            else:
                move = np.argmax(move_probs)        # 最大概率的动作
                self.mcts.update_with_move(-1)
            # =======================================
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
    
    
class Pure_MCTS_Player(object):
    def __init__(self,state,name,policy_value_function, c_puct=5, n_playout=1000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.name = name
        self.last_action = -1
        self.num = 0
        self.state = state

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)  #把-1传进去，就重置了整个树了。

    def get_action(self, state, temp=1e-3, return_prob=1):
        sensible_moves = np.argwhere(state[1]==1)  #可以的选择
        move_probs = np.zeros(12)
        if len(sensible_moves) > 2:
            #=======================================
            # 进行n_playout模拟，生成一棵MCTS，返回根节点的acts, probs
            acts, probs = self.mcts.get_move_probs(self.state, temp, is_pure_tree=1)
            move_probs[list(acts)] = probs

            #=======================================
            #如果是selfplay模式，就要加0.25噪音。然后sample出一个move，执行。
            #如果不是selfplay模式，就不加噪音，但会重置整棵树。
            if self._is_selfplay:
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
            # =======================================
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)