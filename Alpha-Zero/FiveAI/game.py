# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        '''注意这里的参数方式可以这样'''
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board：棋盘位置
        # value: player as pieces type：谁下的棋
        # 更新方式：self.states[move] = self.current_player
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    # 初始化棋盘
    def init_board(self, start_player=0):

        # 判断生成棋盘是否大于胜利条件。否则抛异常。
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))

        # current_player：谁在下棋，初始化的时候为P1
        self.current_player = self.players[start_player]  # start player

        # keep available moves in a list
        # availables：一个list，记录当前可以走棋的位置。某些位置被其他字占据，就不能下了。
        self.availables = list(range(self.width * self.height))
        # 清空状态
        self.states = {}
        # 重置不是最后一步
        self.last_move = -1

    # 把move变成棋盘对应位置。
    # 这里的move相当于action。
    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    # 把棋盘中对应位置转换为move
    def location_to_move(self, location):
        # location是一个二维数组，所以先判断下。如果不是，就返回-1
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        # 如果超出了，就代表不合法，返回-1
        if move not in range(self.width * self.height):
            return -1
        return move

    #state的形式：self.states[move] = self.current_player
    #=============棋盘状态描述================================
    #===1. 棋盘状态用4层，和棋盘大小相同的array进行描述。
    #===No.1 用1.0表示当前这个玩家已经下的子
    #===No.2 用1.0表示另外一个玩家已经下的子
    #===No.3 用1.0表示，当前状态最后一个下的子的位置
    #===No.4 P1玩家所有标1，P2玩家所有标0。
    #===2. Alphazero：用17层。其中16层表示之前两位玩家各8步。
    #========================================================
    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            #============注意这里的操作===============
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            #========================================
            # 第1层：
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            # 第2层：
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # 第3层：
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
            # 第4层：
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    # 执行走棋操作
    def do_move(self, move):
        # 当前玩家下子
        # 1.改变棋盘状态
        # 2.可行的位置去掉这个位置
        # 3.转换下一个下子玩家
        # 4.记录最后一个下子的位置
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    '''
        判断这个状态是否结束了。
        是：返回True，胜利玩家
        否：返回False，-1
    '''
    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    '''
        先判断是否有玩家胜利
        有：True, winner
        否则检查是否还有可以下子的地方
        如果没有：True, -1（-1表示平局）
    '''
    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    # 根据当前状态，画棋盘
    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    # 开始一场游戏
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player) #初始化棋盘

        # 定义board.players：P1=0 P2=1
        # 定义MCTSPlayer；player1和player2
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        #==================开始游戏===============================
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player] #注意：players[]不再只是一个数字，是MCTSPlayer
            move = player_in_turn.get_action(self.board) #该MCTSPlayer会把当前board放入getaction。产出动作move
            self.board.do_move(move) #走棋，交换人，直到结束。
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    #self_play!!!通过selfplay获取数据
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        while True:
            # ======通过get_action，把当前state放入，预测move和概率的列表
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            
            # ======存储s，move_probs，player
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # ======执行动作
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                #如果不是平局：winners_z把winner的步数设置1，否则-1
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                #每一次跑完游戏，都会重新建立一棵树
                player.reset_player()

                #show文字
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            # 其实都只返回1个
            # winner：胜利玩家，1个值（P1 or P2）
            # 假设这个是有N步，那么每一步都产生这样的data
            # states：大小4xHxW，记录state，用于放到network，产生v-preds(一个值) 和 p-preds(每个位置胜率)
            # mcts_probs：HxW，通过建立mcts树，用UCB跑多次，根据访问次数的胜率预计。会作为p-preds更新标准
            # winners_z： 单一值。这个值是通过最后结果反过来填的，如果在这个s下，下棋的人获胜，那么+1，否则-1.平0。用于更新v-preds
