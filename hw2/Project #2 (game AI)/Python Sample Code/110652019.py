import STcpClient_1 as STcpClient
import copy
import numpy as np
import random
import math

def initialMap(node_num, seed):
    # create border
    temp_map = np.ones((14, 14), dtype=np.int32)
    temp_map[1:13, 1:13] = np.zeros([12, 12])

    while True:
        n_free = 0
        t = [[7, 7]]
        prob = 0.7
        random.seed(seed)
        seed += 1
        rand = random.random()
        if rand < prob:
            # as free
            n_free += 1
            temp_map[7][7] = -1
        else:
            n_free = 2
            temp_map[7][7] = 1

        while n_free < node_num:
            if len(t) == 0 & n_free != node_num:
                # recreate
                print("recreate")
                n_free = 0
                temp_map[1:13, 1:13] = np.zeros([12, 12])
                t = [[7, 7]]
                prob = 0.7
                random.seed(seed)
                seed += 1
                rand = random.random()
                if rand < prob:
                    # as free
                    n_free += 1
                    temp_map[7][7] = -1
                else:
                    temp_map[7][7] = 1
                continue
            random.seed(seed)
            seed += 1
            random.shuffle(t)
            x, y = t.pop()
            window = temp_map[x - 1:x + 2, y - 1:y + 2]

            neighbor = []
            # 3
            if window[0][1] == 0:
                neighbor.append([x - 1, y])
            # 4
            if window[2][1] == 0:
                neighbor.append([x + 1, y])

            if y % 2 == 1:
                # 1
                if window[0][0] == 0:
                    neighbor.append([x - 1, y - 1])

                # 2
                if window[1][0] == 0:
                    neighbor.append([x, y - 1])

                # 5
                if window[0][2] == 0:
                    neighbor.append([x - 1, y + 1])

                # 6
                if window[1][2] == 0:
                    neighbor.append([x, y + 1])

            elif y % 2 == 0:
                # 1
                if window[1][0] == 0:
                    neighbor.append([x, y - 1])
                # 2
                if window[2][0] == 0:
                    neighbor.append([x + 1, y - 1])

                # 5
                if window[1][2] == 0:
                    neighbor.append([x, y + 1])
                # 6
                if window[2][2] == 0:
                    neighbor.append([x + 1, y + 1])

            np.random.seed(seed)
            seed += 1
            rand = np.random.random(len(neighbor))
            rand = rand < prob

            for i in range(len(neighbor)):
                m, n = neighbor[i]
                if rand[i]:
                    # as free
                    n_free += 1
                    t.append([m, n])
                    temp_map[m][n] = 1
                else:
                    temp_map[m][n] = -1
                if n_free == node_num: break

        n_component, _, _ = getConnectRegion(1, temp_map[1:13, 1:13])
        if n_component != 1:
            # print('recreate because not 1-component')
            temp_map[1:13, 1:13] = np.zeros([12, 12])
        else:
            break

    # fill all hole
    temp_map[temp_map == 0] = -1

    initMapStat = temp_map[1:13, 1:13]
    initMapStat[initMapStat == 1] = 0

    return initMapStat
def getConnectRegion(targetLabel, mapStat):
    '''

    :param targetLabel:
    :param mapStat:
    :return: numbers of connect region, total occupied area, max connect region
    '''
    # turn into boolean array
    mask = mapStat == targetLabel
    n_field = np.count_nonzero(mask)

    # print(flagArr)

    n_components = 0
    # connection region

    ind = np.where(mask == 1)
    labels = np.zeros((14, 14), dtype=np.int32)
    for k in range(len(ind[0])):
        m, n = ind[0][k], ind[1][k]
        if labels[m + 1][n + 1] != 0:
            continue
        else:
            # haven't have mark
            l_window = labels[m:m + 3, n:n + 3]
            if (l_window == 0).all():
                n_components += 1
                labels[m + 1][n + 1] = n_components
            else:
                mark_pos = np.where(l_window != 0)
                neighbor = np.zeros(1, dtype=np.uint8)

                # connect region
                if n % 2 == 0:
                    for l in range(len(mark_pos[0])):
                        i, j = mark_pos[0][l], mark_pos[1][l]
                        if i == 0:
                            if j == 0:
                                neighbor = np.append(neighbor, l_window[i][j])
                            elif j == 1:
                                neighbor = np.append(neighbor, l_window[i][j])
                            elif j == 2:
                                neighbor = np.append(neighbor, l_window[i][j])
                            else:
                                continue
                        elif i == 1:
                            if j == 0:
                                neighbor = np.append(neighbor, l_window[i][j])
                            elif j == 2:
                                neighbor = np.append(neighbor, l_window[i][j])
                            else:
                                continue
                        elif i == 2:
                            if j == 1:
                                neighbor = np.append(neighbor, l_window[i][j])
                            else:
                                continue
                elif n % 2 == 1:
                    for l in range(len(mark_pos[0])):
                        i, j = mark_pos[0][l], mark_pos[1][l]
                        if i == 0:
                            if j == 1:
                                neighbor = np.append(neighbor, l_window[i][j])
                            else:
                                continue
                        elif i == 1:
                            if j == 0:
                                neighbor = np.append(neighbor, l_window[i][j])
                            elif j == 2:
                                neighbor = np.append(neighbor, l_window[i][j])
                            else:
                                continue
                        elif i == 2:
                            if j == 0:
                                neighbor = np.append(neighbor, l_window[i][j])
                            elif j == 1:
                                neighbor = np.append(neighbor, l_window[i][j])
                            elif j == 2:
                                neighbor = np.append(neighbor, l_window[i][j])
                            else:
                                continue

                neighbor = np.delete(neighbor, 0)
                # mark m,n as min class in the neighborhood
                if neighbor.size == 0:

                    n_components += 1
                    labels[m + 1][n + 1] = n_components
                else:
                    labels[m + 1][n + 1] = min(neighbor)
                    for i in np.unique(neighbor):
                        if i != min(neighbor):
                            # print(f'{i} -> {min(neighbor)}')
                            labels[labels == i] = min(neighbor)

    n_components = len(np.unique(labels)) - 1
    counts = []
    for k in np.unique(labels):
        if k == 0: continue
        c = np.count_nonzero(labels == k)
        counts = np.append(counts, c)
    return n_components, n_field, max(counts)  
def play(player, mapStat, move):
    new_mapStat = copy.deepcopy(mapStat)

    [move_pos_x, move_pos_y] = move[0]  # expected [x,y]
    steps = move[1]  # how many step
    move_dir = move[2]  # 1~6

    next_x = move_pos_x
    next_y = move_pos_y
    new_mapStat[next_x][next_y] = player
    for i in range(steps - 1): 
        [next_x, next_y]=Next_Node(next_x,next_y,move_dir)
        new_mapStat[next_x][next_y] = player
    
    return new_mapStat   
def Next_Node(pos_x,pos_y,direction):
    if pos_y%2==1:
        if direction==1:
            return pos_x,pos_y-1
        elif direction==2:
            return pos_x+1,pos_y-1
        elif direction==3:
            return pos_x-1,pos_y
        elif direction==4:
            return pos_x+1,pos_y
        elif direction==5:
            return pos_x,pos_y+1
        elif direction==6:
            return pos_x+1,pos_y+1
    else:
        if direction==1:
            return pos_x-1,pos_y-1
        elif direction==2:
            return pos_x,pos_y-1
        elif direction==3:
            return pos_x-1,pos_y
        elif direction==4:
            return pos_x+1,pos_y
        elif direction==5:
            return pos_x-1,pos_y+1
        elif direction==6:
            return pos_x,pos_y+1
def checkMoveValidation(player, mapStat, move):
    # move =[move position, move # of step, move direction]
    [pos_x, pos_y] = move[0]  # expected [x,y]
    if (move[1] < 1 or move[1] > 3):
        #print(f"player {player} illegal length.")
        return False
    if mapStat[pos_x][pos_y] != 0:
        #print(f"player {player} illegal place.")
        return False
    next_x, next_y=pos_x,pos_y
    for i in range(move[1] - 1): 
        [next_x, next_y]=Next_Node(next_x,next_y,move[2])
        #print("NEW POS {i}: ", next_x, "----", next_y)
        if(next_x < 0 or next_x > 11 or next_y < 0 or next_y > 11 or mapStat[next_x][next_y]!=0):
            #print(f"player {player} illegal move.")
            return False
    return True
def checkRemainMove(mapStat):
    free_region = (mapStat == 0)
    temp = []
    for i in range(len(free_region)):
        for j in range(len(free_region[0])):
            if(free_region[i][j] == True):
                temp.append([i,j])
    return temp
def getActionSize():
        return 12 * 12 * (6 * 2 + 1)
def getValidMove(player, mapStat):
    valid_move = [0]*getActionSize()
    idx = 0
    for x in range(12):
        for y in range(12):
            if mapStat[x][y] == -1:
                continue
            pos = [x,y]
            act = [pos ,1, 1]
            if checkMoveValidation(player=player, mapStat=mapStat, move=act):
                valid_move[idx] = 1            
            idx += 1
            for l in range(2, 4):
                for dir in range(1, 7):
                    act = [pos, l, dir]
                    if checkMoveValidation(player=player, mapStat=mapStat, move=act):
                        valid_move[idx] = 1
                    idx += 1
    return valid_move

def end_game_check(mapStat):
    return not (mapStat==0).any()

def stringRepresentation(board):
        return board.tostring()

def stringRepresentationReadable(board):
    board_s = "".join(str(square) for row in board for square in row)
    return board_s
class AIGame():
    '''
    mapStat: border=-1, free field=0, player_occupied=player_number(1~2)
    gameStat: sheepStat
    playerStat: position of player
    '''
    def __init__(self, node_num):
        self.idx_action_pair = {}
        self.node_num = node_num

    def getInitBoard(self):
        # return initial board (numpy board)
        seed = random.randint(1, 1000)
        n = random.randint(13, 33)
        board = initialMap(node_num=n, seed=seed)
        return np.array(board)

    def getBoardSize(self):
        # (a,b) tuple
        return (12, 12)

    def getActionSize(self):
        # return number of actions
        return 33 * (6 * 2 + 1)

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        idx = 0
        for x in range(12):
            for y in range(12):
                if board[x][y] == -1:
                    continue
                pos = [x,y]
                act = [pos ,1, 1]
                if idx == action:
                    move = act
                idx += 1
                for l in range(2, 4):
                    for dir in range(1, 7):
                        act = [pos, l, dir]
                        if idx == action:
                            move = act
                        idx += 1
        new_board = play(player=(1 if player == 1 else 2), mapStat=board, move = move)
        return (new_board, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valid_move = [0]*self.getActionSize()
        idx = 0
        for x in range(12):
            for y in range(12):
                if board[x][y] == -1:
                    continue
                pos = [x,y]
                act = [pos ,1, 1]
                if checkMoveValidation(player=player, mapStat=board, move=act):
                    valid_move[idx] = 1            
                idx += 1
                for l in range(2, 4):
                    for dir in range(1, 7):
                        act = [pos, l, dir]
                        if checkMoveValidation(player=player, mapStat=board, move=act):
                            valid_move[idx] = 1
                        idx += 1
        return np.array(valid_move)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if not (board == 0).any():
            if player == 1:
                return 1
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        b = np.copy(board)
        if player == -1:
            for i in range(12):
                for j in range(12):
                    if b[i][j] == 1:
                        b[i][j] = 2
                    elif b[i][j] == 2:
                        b[i][j] = 1
        return board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.getActionSize(board))  # 1 for pass
        #pi_board = np.reshape(pi, (12, 12, 13))
        l = [(board, pi)]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(square for row in board for square in row)
        return board_s

EPS = 1e-8

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, args, nnet=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        #if s not in self.Ps:
        #    # leaf node
        #    self.Ps[s], v = self.nnet.predict(canonicalBoard)
        #    valids = self.game.getValidMoves(canonicalBoard, 1)
        #    self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
        #    sum_Ps_s = np.sum(self.Ps[s])
        #    if sum_Ps_s > 0:
        #        self.Ps[s] /= sum_Ps_s  # renormalize
        #    else:
        #        # if all valid moves were masked make all valid moves equally probable

        #        # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
        #        # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
        #        log.error("All valid moves were masked, doing a workaround.")
        #        self.Ps[s] = self.Ps[s] + valids
        #        self.Ps[s] /= np.sum(self.Ps[s])

        #    self.Vs[s] = valids
        #    self.Ns[s] = 0
        #    return -v

        if s not in self.Ns:
            self.Ns[s] = 0
        valids = self.game.getValidMoves(canonicalBoard, 1)
        
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct  * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

args = dotdict({
    'numIters': 100,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 2,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

class Agent:
    def __init__(self, game, mcts):
        self.game = game
        self.mcts = mcts

    def Getstep(self, mapStat, gameStat):
        #Please write your code here
        action = np.argmax(self.mcts.getActionProb(mapStat, temp=0))
        idx = 0
        for x in range(12):
            for y in range(12):
                if mapStat[x][y] == -1:
                    continue
                pos = [x,y]
                act = [pos ,1, 1]
                if idx == action:
                    move = act
                idx += 1
                for l in range(2, 4):
                    for dir in range(1, 7):
                        act = [pos, l, dir]
                        if idx == action:
                            move = act
                        idx += 1
        return move

print('start game')
agent = Agent(AIGame(33), MCTS(AIGame(33), args))

while (True):

    (end_program, id_package, mapStat, gameStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    
    decision_step = agent.Getstep(mapStat, gameStat)
    
    STcpClient.SendStep(id_package, decision_step)
