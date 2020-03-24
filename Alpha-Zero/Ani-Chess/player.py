import random
import numpy as np
from network import PolicyValueNet

class Player():
    def __init__(self,group,name):
        self.group = group
        self.name = name
    
    def get_action(self,board):
        #ramdom
        #state = board.board_to_state(board.board)
        legal_action = board.get_legal_action(board.board,self.group)
        act_probs = np.random.rand(16,5)#生成随机
        act_probs = act_probs * legal_action
        action = np.where(act_probs==np.max(act_probs))
        return action[0][0],action[1][0]