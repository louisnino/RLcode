import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode():
    def __init__(self ,parent,prior_p):
        self._parent = parent
        self._children = {}
        self._n_visit = 0
        self._q = 0 
        self._u = 0
        self._p = prior_p

    def print_children(self):
        for child in self._children:
            print(child)

    def expand(self,action_probs,game_board):
        legal_action = game_board.get_legal_action(game_board.board)
        #print('EXPAND-player_name:',game_board.current_player.name)
        #print('EXPAND-player_group:',game_board.current_player.group)
        
        #这里需要过滤一下，不能走的位置呀
        #print('expand check')
        #print(legal_action)
        for h in range(action_probs.shape[1]):
            for w in range(action_probs.shape[2]):
                if legal_action[h][w]==0:
                    continue
                else:
                    action = (h,w)
                    #print(action)
                    if action not in self._children:
                        self._children[action] = TreeNode(self,action_probs[0][h][w])
        #self.print_children()

    def get_value(self, c_puct):
        self._u = (c_puct * self._p * np.sqrt(self._parent._n_visit)/(1 + self._n_visit))
        return self._q + self._u

    def select(self,c_puct):
        '''
        获取value最大的node
        '''
        #print('======_children.items======')
        #print(self._children.items())
        return max(self._children.items(),key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visit += 1
        self._q = 1.0 * (leaf_value - self._q)/self._n_visit

    def update_recursive(self,leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10):
        self._root = TreeNode(None,1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, game_board):
        node = self._root
        step_count = copy.deepcopy(game_board.step_count)
        #Select
        while(1):
            if node.is_leaf():
                break
            #node.print_children()
            #game_board.show_board()
            action, node = node.select(self._c_puct)
            #print('SELECT-player_name:',game_board.current_player.name)
            #print('SELECT-player_group:',game_board.current_player.group)
            #print('SELECT-action',action)
            game_board.move(action,change_player=True)
            step_count += 1
            #game_board.show_board()
        #Expand
        action_probs, leaf_value = self._policy(game_board)  #在这个function里面再转state
        end, winner = game_board.get_winner()
        if step_count >= game_board.max_step:
            end, winner = 1,  -1
        if not end :
            node.expand(action_probs,game_board) #这里需要改一下，把game_board传进去，过滤掉不能用的动作
            #print(node)
        else:
            '''
            -1：平局
            0：蓝胜
            1：红胜
            '''
            if winner == -1:#平局
                leaf_value = 0
            elif game_board.current_player == winner:
                leaf_value = 1.0
            else:
                leaf_value = -1.0
        #update
        #print(node)
        node.update_recursive(-leaf_value)

    def get_move_probs(self, game_board,group,temp=1e-3, is_pure_tree=0):
        '''
        开始用mcts计算action
        '''
        for i in range(self._n_playout):
            print('No.%i  play out'%(i))
            game_board_copy = copy.deepcopy(game_board)
            self._playout(game_board_copy)
        action_visit = [(act, node._n_visit) for act, node in self._root._children.items()]
        acts , visit = zip(*action_visit)
        act_porbs = softmax(1.0 / temp * np.log(np.array(visit) + 1e-10))
        return acts, act_porbs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None,1.0)

class MCTS_Player(object):
    def __init__(self, group, name,policy_value_function, c_puct=5, n_playout=10, is_selfplay=1):
        self.group = group
        self.name = name
        self.mcts  = MCTS(policy_value_function,c_puct,n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(-1)
    
    def set_player_ind(self, p):
        self.player = p

    def get_action(self,game_board):
        
        acts, act_porbs = self.mcts.get_move_probs(game_board,self.group, temp=1e-3)
        act_list = []

        for a in acts:
            act_list.append(zip(a))
        if self._is_selfplay:
            move = list(zip(*(np.random.choice(act_list,p=0.75*act_porbs + 0.25*np.random.dirichlet(0.3*np.ones(len(act_porbs)))))))[0]
            print('move====',move)
            self.mcts.update_with_move(move)
        else:
            move = list(zip(*np.random.choice(act_list, p=act_porbs)))[0]
            self.mcts.update_with_move(-1)
        return move 


