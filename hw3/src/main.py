import random
import math
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from itertools import combinations


class Game:
    def __init__(self, difficulty=0, init_num_safe_cell=None):
        '''
        Initialize the game board

        Parameters
        ----------
        difficulty : int
            0: Easy, 1: Medium, 2: Hard

        Returns
        -------
        None
        '''
        board_configurations = [
            (9, 9, 10),   # Easy
            (16, 16, 25), # Medium
            (16, 30, 99)  # Hard
        ]
        self.h, self.w, self.num_of_mines = board_configurations[difficulty] # height, width, number of mines
        self.board = [[0 for _ in range(self.w)] for _ in range(self.h)] # -1: mine, 0~8: number of mines around
        self.shown_cell = [[False for _ in range(self.w)] for _ in range(self.h)] # Indicate the cell is opened or not
        self.mine_pos = set() # The position of mines
       
        # Randomly generate mines
        while len(self.mine_pos) < self.num_of_mines:
            i = random.randrange(self.h)
            j = random.randrange(self.w)
            if (i, j) not in self.mine_pos:
                self.mine_pos.add((i, j))
                self.board[i][j] = -1

        self.init_num_safe_cell = init_num_safe_cell
        num = round(math.sqrt(self.h * self.w)) if init_num_safe_cell is None else init_num_safe_cell
        # num = 10
        self.init_cells = set()
        while len(self.init_cells) < num:
            i = random.randrange(self.h)
            j = random.randrange(self.w)
            if (i, j) not in self.mine_pos and (i, j) not in self.init_cells:
                self.init_cells.add((i, j))


    def open_cell(self, cell, safe):
        '''
        Open the cell and return the number of mines around the cell

        Parameters
        ----------
        cell : tuple
            The position of the cell
        safe : bool
            True if the cell is safe, False if the cell is a mine

        Returns
        -------
        int
            The number of mines around the cell, return -1 if wrongly opened
        '''
        if ((cell in self.mine_pos) ^ (not safe)) or self.shown_cell[cell[0]][cell[1]]:
            return -1
        if cell not in self.mine_pos:
            self.board[cell[0]][cell[1]] = self.get_surround_mines(cell)
        else:
            self.board[cell[0]][cell[1]] = "X"
            
        self.shown_cell[cell[0]][cell[1]] = True

        return self.board[cell[0]][cell[1]]
    
    def get_hint(self, cell):
        '''
        Get the hint of the cell

        Parameters
        ----------
        cell : tuple
            The position of the cell

        Returns
        -------
        list
            The list of the cells around the cell
        int
            The number of mines around the cell
        '''
        cnt = 0
        res = []
        for i in range(cell[0]-1, cell[0]+2):
            for j in range(cell[1]-1, cell[1]+2):
                if i < 0 or i >= self.h or j < 0 or j >= self.w:
                    continue
                if self.shown_cell[i][j]:
                    continue
                if (i, j) != cell:
                    if (i, j) in self.mine_pos:
                        cnt += 1
                    res.append((i, j))
        return res, cnt
    
    def get_surround_mines(self, cell):
        '''
        Get the number of mines around the cell

        Parameters
        ----------
        cell : tuple
            The position of the cell

        Returns
        -------
        int
            The number of mines around the cell
        '''
        cnt = 0
        for i in range(cell[0]-1, cell[0]+2):
            for j in range(cell[1]-1, cell[1]+2):
                if (i, j) in self.mine_pos:
                    cnt += 1
        return cnt

    
    def get_init_safe_cells(self):
        '''
        Get the initial safe cells

        Parameters
        ----------
        None

        Returns
        -------
        set
            The set of the initial safe cells
        '''
        
        
        return self.init_cells

    def print_board(self):
        '''
        Print the game board. ? means the cell is not opened yet. X means the cell is a mine. 0~8 means the number of mines around the cell.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        for i in range(self.h):
            for j in range(self.w):
                if self.shown_cell[i][j]:
                    print(self.board[i][j], end=' ')
                else:
                    print('?', end=' ')
            print()

class Literal:
    '''
    A literal is a cell with a positive or negative sign. For example, (0, 0) is a positive literal, and (0, 0)' is a negative literal.
    '''
    def __init__(self, cell, is_posi):
        '''
        Initialize the literal

        Parameters
        ----------
        cell : tuple
            The position of the cell
        is_posi : bool
            True if the literal is positive, False if the literal is negative

        Returns
        -------
        None       
        '''
        self.cell = cell
        self.posi = is_posi

    def __eq__(self, other):
        '''
        Check if two literals are the same
        '''
        return self.cell == other.cell and self.posi == other.posi
    
    def __str__(self):
        '''
        Return the string of the literal
        '''
        return str(self.cell) + ('' if self.posi else "'")
    
    def __hash__(self):
        '''
        Return the hash value of the literal
        '''
        return hash(str(self))
    
class Clause:
    '''
    A clause is a set of literals
    '''
    def __init__(self, literals=[]):
        '''
        Initialize the clause

        Parameters
        ----------
        literals : list
            The list of literals
        
        Returns
        -------
        None
        '''
        self.literals = set(literals)
        
    
    def __str__(self):
        '''
        Return the string of the clause
        '''
        return "[" + ' '.join([str(l) for l in self.literals]) + "]"
    
    def __eq__(self, other):
        '''
        Check if two clauses are the same
        '''
        return self.literals == other.literals
    
    def __len__(self):
        '''
        Return the number of literals in the clause
        '''
        return len(self.literals)
    
    def __hash__(self):
        '''
        Return the hash value of the clause
        '''
        return hash(str(self))
    
    def __copy__(self):
        '''
        Return the copy of the clause
        '''
        return Clause(self.literals.copy())
    
    
class KB:
    '''
    A knowledge base is a set of clauses
    '''
    def __init__(self, clauses=set()):
        '''
        Initialize the knowledge base

        Parameters
        ----------
        clauses : set
            The set of clauses

        Returns
        -------
        None
        '''
        self.clauses = clauses

    def insert(self, clause: Clause, KB0):
        '''
        Insert a clause into the knowledge base

        Parameters
        ----------
        clause : Clause
            The clause to be inserted
        KB0 : KB
            The knowledge base that contains of claueses that are already inferred

        Returns
        -------
        None
        '''
        for clause1 in KB0.clauses:
            cell_pos = list(clause1.literals)[0].cell
            pos = list(clause1.literals)[0].posi
            for lit in clause.literals.copy():
                if lit.cell == cell_pos and lit.posi != pos and lit in clause1.literals:
                    clause.literals.remove(lit)
        if len(clause.literals) == 0:
            return None
        if clause in self.clauses:
            return None
        for clause1 in self.clauses.copy():
            if clause1.literals.issubset(clause.literals):
                return None
            elif clause.literals.issubset(clause1.literals):
                if clause1 in self.clauses:
                    self.clauses.remove(clause1)
        if clause in KB0.clauses or clause in self.clauses or len(clause.literals) == 0:
            return None
        
        if len(clause.literals) >= 1:
            self.clauses.add(clause)
        # print(f"insert {clause}")
        # print(f"[\n{','.join([str(c) for c in self.clauses])}\n]")

def matching_clauses(a: Clause, b: Clause):
    '''
    Check if two clauses can be matched using resolution

    Parameters
    ----------
    a : Clause
        The first clause
    b : Clause
        The second clause

    Returns
    -------
    a : Clause
        The first clause after matching
    b : Clause
        The second clause after matching
    '''
    if len(a) > 2 and len(b) > 2:
        return a, b
    
    if a == b:
        return a, None
    
    if a.literals.issubset(b.literals):
        return a, None
    
    if b.literals.issubset(a.literals):
        return b, None
    
    a = Clause(a.literals.copy())
    b = Clause(b.literals.copy())
    
    comps = set()
    for i in a.literals:
        for j in b.literals:
            if i.cell == j.cell and i.posi != j.posi:
                comps.add(i.cell)
    
    if len(comps) == 1:
        a = Clause(list(a.literals.copy().union(b.literals.copy())))
        b = None
        for i in comps:
            if Literal(i, True) in a.literals:
                a.literals.remove(Literal(i, True))
            if Literal(i, False) in a.literals:
                a.literals.remove(Literal(i, False))
        return a, None

    return (a if len(a.literals) > 0 else None), (b if len(b.literals) > 0 else None)
    

class Player:
    '''
    The player class
    '''
    def __init__(self, game: Game):
        '''
        Initialize the player

        Parameters
        ----------
        game : Game
            The game to be played

        Returns
        -------
        None
        '''
        self.game = game
        self.KB = KB(set())
        self.KB0 = KB(set())
        self.mine = set()
        self.safe = set()
        for i in self.game.get_init_safe_cells():
            self.safe.add(i)
            self.KB.insert(Clause([Literal(i, False)]), self.KB0)

    def play(self, silent=False):
        '''
        Play the game

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        unmarked_cnt = 0
        while unmarked_cnt <= 10:
            if not silent:
                os.system('cls')
                self.game.print_board()
                print(f"# in KB: {len(self.KB.clauses)}, # in KB0: {len(self.KB0.clauses)}")
                print(f"# single clause in KB: {len([clause for clause in self.KB.clauses if len(clause) == 1])}")
                # for clause in self.KB.clauses:
                #     print(clause, len(clause))
                # print("----")
                # for clause in self.KB0.clauses:
                #     print(clause)
            updated = False
            if Clause([]) in self.KB.clauses:
                self.KB.clauses.remove(Clause([]))
            for clause in self.KB.clauses:
                if len(clause) == 1:
                    unmarked_cnt = 0
                    updated = True
                    lit = list(clause.literals)[0]
                    self.KB.clauses.remove(clause)
                    self.KB0.clauses.add(Clause(clause.literals.copy()))
                    if not silent:
                        print(f"Open cell {lit.cell} with {lit.posi}")
                    if lit.posi:
                        if self.game.open_cell(lit.cell, False) == -1:
                            print('Game Over!')
                            return -1
                        self.mine.add(lit.cell)
                    else:
                        if self.game.open_cell(lit.cell, True) == -1:
                            print('Game Over!')
                            return -1
                        self.safe.add(lit.cell)
                    # for clause1 in self.KB.clauses.copy():
                    #     print(clause1)
                    for clause1 in self.KB.clauses.copy():
                        if clause1 in self.KB.clauses:
                            self.KB.clauses.remove(clause1)
                        a, b = matching_clauses(Clause(clause.literals.copy()), clause1)
                        if a:
                            self.KB.insert(a, self.KB0)
                        if b:
                            self.KB.insert(b, self.KB0)
                    
                    if not lit.posi:
                        pos, n = self.game.get_hint(lit.cell)
                        # print(pos, n)
                        if len(pos) == n:
                            for i in pos:
                                self.KB.insert(Clause([Literal(i, True)]), self.KB0)
                        elif n == 0:
                            for i in pos:
                                self.KB.insert(Clause([Literal(i, False)]), self.KB0)
                        else:
                            for comb in combinations(pos, len(pos)-n+1):
                                lits = []
                                for cell in comb:
                                    lits.append(Literal(cell, True))
                                self.KB.insert(Clause(lits), self.KB0)
                            for comb in combinations(pos, n+1):
                                lits = []
                                for cell in comb:
                                    lits.append(Literal(cell, False))
                                self.KB.insert(Clause(lits), self.KB0)
                    break
            if updated:
                continue
            len_2_KB_clause = [x for x in list(self.KB.clauses.copy()) if len(x) == 2]
            KB_clause = list(self.KB.clauses.copy())
            if silent:
                print(f"{len(KB_clause)}")
            print("entering pairwise matching")
            unmarked_cnt += 1
            for i in tqdm(len_2_KB_clause):
                for j in KB_clause:
                    if i == j:
                        continue
                    if i in self.KB.clauses:
                        self.KB.clauses.remove(i)
                    if j in self.KB.clauses:
                        self.KB.clauses.remove(j)
                    if(len(i) == 0 or len(j) == 0):
                        continue
                    a, b = matching_clauses(i, j)
                    
                    if a:
                        self.KB.insert(a, self.KB0)
                        if a != i and a != j:
                            # print(len(i), len(j), i, j, a, b)
                            updated = True
                    else:
                        # print(len(i), len(j), i, j, a, b)
                        updated = True
                    if b:
                        self.KB.insert(b, self.KB0)
                        if b != j and b != i:
                            # print(len(i), len(j), i, j, a, b)
                            updated = True
                    else:
                        # print(len(i), len(j), i, j, a, b)
                        updated = True
            
            if not updated:
                if len(self.KB0.clauses) != self.game.h * self.game.w:
                    print("Stuck")
                    return self.game.h * self.game.w - len(self.KB0.clauses)
                else:
                    print("Win!")
                    return 0
        print("Stuck")
        return self.game.h * self.game.w - len(self.KB0.clauses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", 
        "--difficulty", 
        dest="difficulty",
        type=int, 
        help="difficulty level of the game 0: easy, 1: medium, 2: hard", 
        default=0
    )
    parser.add_argument(
        "-n", 
        dest="n",
        type=int, 
        help="number of games to play",
        default=1
    )
    # disable print board
    parser.add_argument(
        "-s",
        "--silent",
        dest="silent",
        action="store_true",
        help="disable print board",
        default=False
    )
    parser.add_argument(
        "-i",
        "--init",
        dest="init",
        help="number of init safe cell",
        type=int,
        default=None
    )
    args = parser.parse_args()
    records = [] # win, time
    random.seed(time.time())
    for i in range(args.n):
        game = Game(args.difficulty, args.init)
        player = Player(game)
        start = time.time()
        res = player.play(args.silent)
        end = time.time()
        records.append((res, end-start))
        print(f"Gmae {i+1} finished in {end-start} sec. with result {res}")
    # Print status with windows
    print("================= Status =================")
    print(f"Play {args.n} games with difficulty {args.difficulty}")
    print(f"Win: {len([i for i in records if i[0] == 0])}")
    print(f"Lose: {len([i for i in records if i[0] == -1])}")
    if len([i for i in records if i[0] > 0]) > 0:
        print(f"Stuck: {len([i for i in records if i[0] > 0])}, avg stuck steps: {round(sum([i[0] for i in records if i[0] > 0])/len([i for i in records if i[0] > 0]), 2)}")
    print(f"Average time: {round(sum([i[1] for i in records])/len(records), 2)} sec.")
    print(f"Std time: {round(np.std([i[1] for i in records]), 2)} sec.")
    print("==========================================")
