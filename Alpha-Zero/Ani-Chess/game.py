import random
import numpy as np
import copy
from player import Player
from network import PolicyValueNet
from mcts import MCTS_Player


class Chess():
    def __init__(self, group, cid, is_show=0):
        self.group = group
        self.cid = cid   #chessid 根据id换算名字，但不分阵营
        self.is_show = is_show
        self.name = self.id_to_name()
        
    def id_to_name(self):
        name = ['象','狮','虎','豹','狼','狗','猫','鼠']
        return name[self.cid]
        

                
class Board():
    
    def __init__(self,mcts_player):
        self.width = 4
        self.height = 4
        self.state = np.zeros([self.height*self.width,2*8+1])
        self.max_step = 200 #最大步数
        #self.step_count = 0 #计数目前走了多少步
        self.mcts_player = mcts_player
        #self.p2 = p2
        self.board = self.init_board() #board是一个list，记载所有的棋子,不维护一个4*4的array了。
        self.dead_chess_list = []

    # 判断是否已经结束,返回winner
    def get_winner(self):
        '''
        -1：平局
        0：蓝胜
        1：红胜
        '''
        group_list = []
        # board中的棋子小于等于1个。（1个的时候是None）
        # 平局
        if len(set(self.board))<=1:
            #print('没有棋子了，平局')
            return 1,-1

        #超出步数
        #平局
        elif self.step_count>=self.max_step:
            #print('超出步数，平局')
            return 1,-1

        #逐个看每个剩余棋子的group
        else:
            for i in range(len(self.board)):
                if self.board[i]==None:
                    continue
                else:
                    group_list.append(self.board[i].group)
            #print('group_list:',group_list)
            if len(set(group_list))<=1:
                if group_list[0]==0:  #如果剩余是蓝，就是蓝胜，否则：红胜
                    #print('剩余棋子，蓝胜')
                    return 1, 0 
                else:
                    #print('剩余棋子，红胜')
                    return 1, 1
            else:
                #print('尚未结束')
                return 0,-1  #没结束

    def init_board(self):
        board_list = []
        for i in range(2):
            for j in range(8):
                chess = Chess(group=i,cid=j,is_show=0)
                board_list.append(chess)
        random.shuffle(board_list)
        self.current_player = 0
        return board_list
    
    def reset_board(self):
        self.board = self.init_board()
        self.step_count = 0
     
    #https://www.cnblogs.com/daofaziran/p/9015284.html
    #python输出带颜色字体详解
    def show_board(self):
        
         for i in range(self.height*self.width):
            # 如果为空
            if self.board[i] == None:
                print('\033[1;37;47m  \033[0m',end='')
            # 是否已经揭开
            elif self.board[i].is_show == 0:
                print('\033[1;37;40m? \033[0m',end='')
            # 判断阵营
            elif self.board[i].group == 0:
                print('\033[1;34;47m%s\033[0m'%(self.board[i].name),end='')
            else:
                print('\033[1;31;47m%s\033[0m'%(self.board[i].name),end='')
            
            if (i+1)%4 ==0:
                print('')  

                    
    #从board转换为当前state
    def board_to_state(self,board):   
        '''
        根据棋子来的，每一层代表一个棋子：
        全-1：未知
        全0：死了
        已知：对应点标记位置
        '''

        state = np.ones([(2*8+1),self.height,self.width]) * (-1) #见设置全部未知，然后根据board里面的恢复

        # 先搜索棋盘上的棋子
        for i in range(len(board)):
            # 那个位置是空的
            if board[i] == None:
                continue
            else:
                index = board[i].group*8 + board[i].cid #找出该棋子应该在第几channel标记
                state[index] = np.zeros([self.width,self.height])
                
                if board[i].is_show == 1:
                    h = i // self.width
                    w = i % self.width
                    #print('index',index)
                    state[index][h,w] = 1

        #搜索死了的棋子
        for j in range(len(self.dead_chess_list)):
            index = self.dead_chess_list[j].group*8 + self.dead_chess_list[j].cid
            state[index] = np.zeros([self.width,self.height])

        #如果是p1,则用0表示，否则用1
        if self.current_player == 0 :
            state[-1] *= 0
        else:
            state[-1] *= -1
        state = np.transpose(state,(1,2,0))
        return state


    #判断两个棋子大小
    def compare_chess(self,chess_a, chess_b):
        '''
            前者胜：1
            平：0
            前者败：-1
        '''
        #print(chess_a)
        a = chess_a.cid
        b = chess_b.cid        
        if a==7 and b==0:
            return 1
        elif b==7 and a==0:
            return -1
        elif a==b:
            return 0
        elif a<b:
            return 1
        else:
            return -1

    #翻转棋子
    def roll(self,pos):
        chess = self.board[pos]
        if chess.is_show==0:
            chess.is_show=1
        return
        
        
    #上下左右
    def move(self,aciton,change_player):
        pos,direct = aciton
        # 从action_space看是否合法，不合法直接返回
        # 这里拆开一下，判断是否合法，在getaction处理，这里只处理走动
        # 如果合法，看看目标是否有敌方
        # 目标位置是敌方，compare下，如果赢了可以行，对方消失；平双方消失,输，自己消失
        
        #如果是=0，直接翻棋
        if direct==0:
            self.roll(pos)
        #如果不是，就看看目标位置和当前位置。
        # 如果目标位置是空，则可以走
        # 否则：吃
        #注意：这里没有判断目标位置是否为自己的棋，因为自己棋应该是非法的。
        else:
            target_pos = self.get_target_pos(pos,direct)
            #如果是空位，那么可以走
            self_chess = self.board[pos]
            if self.board[target_pos] == None:
                self.board[target_pos] = self_chess
                self.board[pos] = None
            else:   
                oppo_chess = self.board[target_pos]
                result = self.compare_chess(self_chess,oppo_chess)

                #赢了
                if result == 1:
                    self.dead_chess_list.append(copy.deepcopy(self.board[target_pos]))
                    self.board[target_pos] = self_chess
                    self.board[pos] = None
                #平
                elif result == 0:
                    self.dead_chess_list.append(copy.deepcopy(self.board[target_pos]))
                    self.dead_chess_list.append(copy.deepcopy(self.board[pos]))
                    self.board[target_pos] = None
                    self.board[pos] = None
                #输
                elif result ==-1:
                    self.dead_chess_list.append(copy.deepcopy(self.board[pos]))
                    self.board[pos] = None

        #self.step_count+=1

        if change_player:
            #print('move-change-player')
            self.change_player()
            
    
    def get_target_pos(self,pos,direct):
        '''
        direct:
        1:上
        2:下
        3:左
        4:右
        返回目标位置id，如果撞墙，就返回-1
        '''
        if direct==1:
            target_pos = pos - 4
            if target_pos<0:
                return -1
            else:
                return target_pos
        elif direct==2:
            target_pos = pos + 4
            if target_pos>15:
                return -1
            else:
                return target_pos
        elif direct==3:
            if pos%4 == 0: #最左的一列
                return -1
            else:
                return pos - 1
            
        elif direct==4:
            if pos%4 == 3: #最右的一列
                return -1
            else:
                return pos + 1  
        else:
            print('非法方向')
    
    
    #从state获取可以移动空间,可移动为1，否则0
    def get_legal_action(self,board):
        action_space = np.zeros([self.width*self.height,5])
        for i in range(len(board)):
            #如果这个位置没棋子，直接跳过
            if board[i]==None:
                continue
                
            # 如果未翻，则可以翻
            if not board[i].is_show:
                action_space[i][0]=1
                continue
                
            #检查阵营是否一致,如果不是自己棋，直接set0,跳出
            elif board[i].group != self.current_player:
                action_space[i] = np.zeros_like(action_space[i])
                continue
                
            #如果是自己棋
            for j in range(1,5):
                if self.get_target_pos(i,j) != -1:
                    target_pos = self.get_target_pos(i,j)                 
                    # 如果是空位，能移动
                    if self.board[target_pos] == None :
                        action_space[i][j] = 1
                    # 如果是自己的棋和未知的棋，就不能动
                    elif self.board[target_pos].is_show == 0 or self.board[target_pos].group == self.current_player:
                        action_space[i][j] = 0
                    # 是敌方的棋
                    elif self.board[target_pos].group != self.current_player:
                        action_space[i][j] = 1
                    else:
                        action_space[i][j] = 0
        
        return action_space


    def change_player(self):
        if self.current_player == 0:
            self.current_player = 1
        else:
            self.current_player = 0



class Game():
    def __init__(self,policy_value_net):
        
        self.policy_value_net = policy_value_net
        #self.p1 = Player(group=0,name = 'p1')  #蓝
        #self.p2 = Player(group=1,name = 'p2')  #红
        self.mcts_player = MCTS_Player(group=0,name = 'p1',policy_value_function = self.policy_value_net.policy_value_fn)  #蓝
        #self.p2 = MCTS_Player(group=1,name = 'p2',policy_value_function = self.policy_value_net.policy_value_fn)  #红
        self.game_board = Board(self.mcts_player)
        self.state_buffer = []
        

        
    def init_player_group(self,pos):
        chess = self.game_board.board[pos]
        if chess.group != 0:
            self.game_board.mcts_player.group = 1
        
    '''
    def start_game(self,is_show_board):
        #board = self.game_board.reset_board()
        self.game_board.show_board()
        print('=======================')
        for i in range(self.game_board.max_step):
            state = self.game_board.board_to_state(self.game_board.board)
            self.state_buffer.append(copy.deepcopy(state))
            pos,dire= self.game_board.current_player.get_action()  #action是一个元组，（pos，移动）
            #print('%s , action: (%i, %i)'%(self.current_player.name,pos,dire))
            
            self.game_board.move(pos,dire)
            if is_show_board:
                self.game_board.show_board()
            
            #第一轮改变一下阵营，p1总是先手，但翻到什么棋，就是什么阵营
            if i == 0 :
                self.init_player_group(pos) 
                
            # 判断是否结束  
            if self.game_board.get_winner()==1:
                break
            else: 
                self.game_board.change_player()
            print('==========step %i=========='%i)
            
        has_winner , winner = self.game_board.get_winner()

        if has_winner==0:
            print('draw')
        else:
            if winner == 0:
                print('winner is p%i'%(winner+1))
            else:
                print('winner is p%i'%(winner+1))
    '''


    def move_to_act(self,move):
        pos,dire = move
        action = np.zeros((16,5))
        action[pos][dire] = 1
        return action


    def start_self_play(self,is_show_board=1):
        
        #定义buffer
        state_buff, action_buff= [], []
        player_group_list = []   #记录走棋的顺序
        self.game_board.reset_board()
        self.mcts_player.reset_player()
        
        #开始游戏：
        for i in range(self.game_board.max_step):
            print('==========step %i=========='%i)
            #self.game_board.show_board()

            #从game_board转为state，并保存。
            state = self.game_board.board_to_state(self.game_board.board)
            state_buff.append(copy.deepcopy(state))   #记录state

            #从game_board计算action，并move
            player_group_list.append(copy.deepcopy(self.game_board.current_player)) #记录下棋顺序
            move = self.game_board.mcts_player.get_action(self.game_board)  #action是一个元组，（pos，移动）
            self.game_board.move(move,change_player=True)

            self.game_board.step_count += 1
            #print('setpcountprint:',self.game_board.step_count)

            #move转化为action的方式，并保存
            action = self.move_to_act(move)
            action_buff.append(action)         #记录action

            #是否要show过程
            if is_show_board:
                self.game_board.show_board()
            
            #第一轮改变一下阵营，p1总是先手，但翻到什么棋，就是什么阵营
            if i == 0 :
                pos,_ = move
                self.init_player_group(pos) 
            
            

            # 判断是否结束 
            end, winner = self.game_board.get_winner()
            #如果结束，
            if end:
                winner_z = np.zeros(len(player_group_list))
                if winner != -1:  #非平局
                    winner_z[np.array(player_group_list)==winner] = 1.0
                    winner_z[np.array(player_group_list)!=winner] = -1.0
                self.mcts_player.reset_player()
                #self.p2.reset_player()
                print('winner:  ',winner)
                #print(winner_z)
                #print(player_group_list)
                break
            #else: 
                #self.game_board.change_player()


        #如果超过步数，则当做打平，并结束
        if i >=self.game_board.max_step:
            winner_z = np.zeros(len(player_group_list))
            self.mcts_player.reset_player()
            print('winner:  draw')
            #print(winner_z)
           

        return zip(state_buff,action_buff,winner_z) 