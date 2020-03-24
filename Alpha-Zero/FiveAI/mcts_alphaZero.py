# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

#定义节点
class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent   #父节点
        self._children = {}     # 子节点，是一个字典：字典的key是动作，item是子节点。子节点包括了描述这个动作的概率，Q等
        self._n_visits = 0      # 记录这个节点被访问次数
        self._Q = 0             #这个节点的价值
        self._u = 0             #用于计算UCB上限。在select的时候，用的是Q+U的最大值。
        self._P = prior_p       #动作对应的概率

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        展开：当一个节点是叶子节点的时候，需要被展开。
            输入action_priors：包含action和对应的概率
            判断这个动作是否在_children的字典中。如果不在，增加这个动作，并增加对应的节点，把概率写在节点中。
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        选择：选择UCB最大的值：UCB = Q(s,a) + U(s,a)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        做一次模拟，把返回的leaf_value去修改Q
        1._n_visits增加
        2.leaf_value和原来的Q，用_n_visits平均一下。1.0是学习率
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

def update_recursive(self, leaf_value):
    """Like a call to update(), but applied recursively for all ancestors.
    用leaf_value反向更新祖先节点。
    因为整棵树，是双方轮流下子的。所以对于一个state update是正的，那么这个state前后updata的数值就是负的。
    """
    # If it is not root, this node's parent should be updated first.
    if self._parent:
        self._parent.update_recursive(-leaf_value)
    self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        UCB = Q(s,a) + U(s,a)
        """
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

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0) #初始化根节点
        self._policy = policy_value_fn #用于生成子节点action-prob对
        self._c_puct = c_puct  #一个常数，好像没啥用
        self._n_playout = n_playout   #模拟多少次走一步

    #进行一次模拟_root就代表传入state
    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            # Greedily select next move.
            #找出UCB最大的动作，并执行。
            action, node = node.select(self._c_puct)
            state.do_move(action)
            #直到去到叶子节点
            if node.is_leaf():
                break

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        # 我们评估这个叶子节点的Q，和他的action-probs
        # 如果还没有结束，那么就扩展这棵树。action-probs放进子节点。
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)

        #如果结束了。
        # 如果平局，就设置成leaf_value = 0
        # 否则: 如果胜利者是当前的，那么leaf_value = 1， 否则leaf_value = -1
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
        # 向上更新祖先节点。
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)


    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        #每一步进行_n_playout次模拟
        #每次都把现在的state复制出来。进行模拟，一直到游戏结束
        #把得到的leaf_value更新到每个状态。同时更新被访问次数。
        #最后我们会得到一颗模拟出来各种结果的树，我们需要的就是这个树。
        for n in range(self._n_playout):
            #关于copy.deepcopy(state)
            #https://blog.csdn.net/u010712012/article/details/79754132
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        # _root._children.items()访问根节点的_children，就是访问当前状态下，各个动作和对应的节点。
        # 取出节点和被访问次数
        # 然后一轮运算后，根据访问次数，获得act 和对应的act_probs
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        #下棋后，检查这move是否在这个树的子节点中。如果在就把根节点移动到这个节点。
        #否则新建一个节点。
        #这棵树会一直维护，直到一次游戏结束。
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        # 输入-1，重置整棵树
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1) #把-1传进去，就重置了整个树了。

    def get_action(self, board, temp=1e-3, return_prob=0):
        #============================================================================
        #进行一次游戏，每一步都会get一次action的。
        #1.首先获取合法动作位置
        #2.基于当前状态，进行_n_playout次模拟。生成树，并返回acts, probs(注意：这个prob是根据树的访问次数来的，不是通过network来的)
        #3.如果是selfplay模式，那么加噪音然后sample。然后挪动树的根节点。（树是保留的）
        #  如果不是selfplay模式，那么不加噪音。 重置整棵树
        # ============================================================================
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)

        if len(sensible_moves) > 0:
            # 进行n_playout模拟，生成一棵MCTS，返回根节点的acts, probs
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs

            #=======================================
            #如果是selfplay模式，就要加0.25噪音。然后sample出一个move，执行。
            #如果不是selfplay模式，就不加噪音，但会重置整棵树。
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)            #把树的根节点和当前状态对应。
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
                #location = board.move_to_location(move)
                #print("AI move: %d,%d\n" % (location[0], location[1]))
            # =======================================
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
