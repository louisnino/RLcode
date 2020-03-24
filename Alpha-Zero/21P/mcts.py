import numpy as np
import copy

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

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent  #父节点
        self._children = {}  # 子节点，是一个字典：字典的key是动作，item是子节点。子节点包括了描述这个动作的概率，Q等
        self._n_visits = 0   # 记录这个节点被访问次数
        self._Q = 0    #这个节点的价值
        self._u = 0    #用于计算UCB上限。在select的时候，用的是Q+U的最大值。
        self._P = prior_p   #动作对应的概率
        #self.curr_player = curr_player #要不要传入一个s，或者player比较好？
           
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
                
    def select(self, c_puct):
        return max(self._children.items(),key=lambda act_node: act_node[1].get_value(c_puct))
    
    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
    
    # 从祖先一直下来，按着pointer刚更新
    def update_recursive(self, leaf_value_buffer, pointer):
        if self._parent:
            self._parent.update_recursive(leaf_value_buffer, pointer)
        self.update(leaf_value_buffer[pointer])
        pointer +=1
    
    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                    np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
    
    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=500):
        self._root = TreeNode(None, 1.0)    #初始化根节点
        self._policy = policy_value_fn      #用于生成子节点action-prob对
        self._c_puct = c_puct               #一个常数，好像没啥用
        self._n_playout = n_playout         #模拟多少次走一步

    
    def _playout(self, state):
        '''
        主要功能就是创建一颗MCTS。
        '''
        #进行一次模拟_root就代表传入state
        node = self._root           #把这个node变成当前根节点
        run_down = []               #存player的顺序 

        #Select--expand---updata
        while(1):            
            if node.is_leaf():      #如果已经是叶子节点，就不需要
                break

            action, node = node.select(self._c_puct)    # select：选择节点:选择最大分数的
            curr_player = state.get_curr_player()       # 记录当前玩家
            run_down.append(copy.copy(curr_player))
            state.do_move(action)                       # 进行下一步，直到已经到叶子节点。

        action_probs, leaf_value = self._policy(state)  # 【敲黑板】用网络预估动作的概率和叶子的价值。


        if count_one(state.current_state[1],1)>2:       # 判断游戏是否应该结束了
            curr_player = state.get_curr_player()       
            node.expand(action_probs)                   # 如果没有，就扩展叶子节点。  

            # ======update   
            # leaf_value_buffer 是用于反向更新
            pointer = 0
            leaf_value_buffer = []
            if len(run_down)>0:             # 当只有根节点的时候。
                for player in run_down:
                    if player == curr_player:
                        leaf_value_buffer.append(leaf_value)
                    else:
                        leaf_value_buffer.append(-leaf_value) 
                # 向上更新祖先节点。
                node.update_recursive(leaf_value_buffer,pointer)
        
        # 如果游戏结束了，就算出winner
        else:
            if state.current_state[2][0]<=21 and state.current_state[3][0]<=21:
                if state.current_state[2][0]>=state.current_state[3][0]:
                    winner = 0
                else:
                    winner = 1
            elif state.current_state[3][0]>21:
                winner = 0
            else:
                winner = 1

            #====update====
            pointer = 0
            leaf_value_buffer = []
            for player in run_down:
                if player == winner:
                    leaf_value_buffer.append(1)
                else:
                    leaf_value_buffer.append(-1)
            #向上更新祖先节点。
            node.update_recursive(leaf_value_buffer,pointer)


            
    #===============_puretree=========================================================
    def _puretree_expand_fn(self,state):
        legal_position = get_avail_act(state.current_state[1])
        act_porbs = zip(legal_position,(np.ones(12)/12))
        return act_porbs, 1
        
    #进行一次模拟_root就代表传入state（不用network，纯用树playout）
    def _puretree_playout(self, state):
        node = self._root #把这个node变成当前根节点
        run_down = []  #存player的顺序        
        #Select--expand---updata
        while(1): 
            
            if node.is_leaf():
                break                
            action, node = node.select(self._c_puct)
            curr_player = state.get_curr_player()
            run_down.append(copy.copy(curr_player))
            state.do_move(action) 
            
        #action_probs, leaf_value = self._policy(state)
        action_probs,leaf_value = self._puretree_expand_fn(state)
        if count_one(state.current_state[1],1)>2:
            curr_player = state.get_curr_player()
            node.expand(action_probs)
            # ======update   
            # leaf_value_buffer 是用于反向更新
            pointer = 0
            leaf_value_buffer = []
            if len(run_down)>0:             # 当只有根节点的时候。
                for player in run_down:
                    if player == curr_player:
                        leaf_value_buffer.append(leaf_value)
                    else:
                        leaf_value_buffer.append(-leaf_value) 
                # 向上更新祖先节点。
                node.update_recursive(leaf_value_buffer,pointer)    
                  
        else:
            if state.current_state[2][0]<=21 and state.current_state[3][0]<=21:
                if state.current_state[2][0]>=state.current_state[3][0]:
                    winner = 0
                else:
                    winner = 1
            elif state.current_state[3][0]>21:
                winner = 0
            else:
                winner = 1
            
            pointer = 0
            leaf_value_buffer = []
            for player in run_down:
                if player == winner:
                    leaf_value_buffer.append(1)
                else:
                    leaf_value_buffer.append(-1)
            #向上更新祖先节点。
            node.update_recursive(leaf_value_buffer,pointer)
                       
    def get_move_probs(self, state, temp=1e-3, is_pure_tree=0):
        #每一步进行_n_playout模拟
        #每次都把现在的state复制出来。进行模拟，一直到游戏结束
        #把得到的leaf_value更新到每个状态。同时更新被**访问次数**。
        #最后我们会得到一颗模拟出来各种结果的树，我们需要的就是这个树。
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)           #复制当前的state出来，
            if is_pure_tree:                            #如果是纯树，就用纯树playout
                self._puretree_playout(state_copy)
            else:                                       #用MCTS的playout，建立一颗MCTS
                self._playout(state_copy)

        # 【敲黑板】这里是通过节点的访问次数，返回的动作和动作概率
        # _root._children.items()访问根节点的_children，就是访问当前状态下，各个动作和对应的节点。
        # 取出节点和被访问次数
        # 然后一轮运算后，根据访问次数，获得act 和对应的act_probs
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        #下棋后，检查这move是否在这个树的子节点中。如果在就把根节点移动到这个节点。
        #否则新建一个节点。
        #这棵树会一直维护，直到一次游戏结束。
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


    def __str__(self):
        return "MCTS"


