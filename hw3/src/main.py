import random
import math
import os
import argparse
from tqdm import tqdm
from itertools import combinations


class Game:
    def __init__(self, difficulty=0):
        board_configurations = [
            (9, 9, 10),
            (16, 16, 25),
            (16, 30, 99)
        ]
        self.h, self.w, self.num_of_mines = board_configurations[difficulty]
        self.board = [[0 for _ in range(self.w)] for _ in range(self.h)]
        self.shown_cell = [[False for _ in range(self.w)] for _ in range(self.h)]
        self.mine_pos = set()
        self.found_mines = set()

        while len(self.mine_pos) < self.num_of_mines:
            i = random.randrange(self.h)
            j = random.randrange(self.w)
            if (i, j) not in self.mine_pos:
                self.mine_pos.add((i, j))
                self.board[i][j] = -1

    def open_cell(self, cell, safe):
        if ((cell in self.mine_pos) ^ (not safe)) or self.shown_cell[cell[0]][cell[1]]:
            return -1
        if cell not in self.mine_pos:
            self.board[cell[0]][cell[1]] = self.get_surround_mines(cell)
        else:
            self.board[cell[0]][cell[1]] = "X"
            
        self.shown_cell[cell[0]][cell[1]] = True

        return self.board[cell[0]][cell[1]]
    
    def cell_status(self, cell):
        return self.board[cell[0]][cell[1]]
    
    def get_hint(self, cell):
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
        cnt = 0
        for i in range(cell[0]-1, cell[0]+2):
            for j in range(cell[1]-1, cell[1]+2):
                if (i, j) in self.mine_pos:
                    cnt += 1
        return cnt

    def is_mine(self, cell):
        return cell in self.mine_pos
    
    def check_win(self):
        return self.found_mines == self.mine_pos
    
    def get_init_safe_cells(self):
        num = round(math.sqrt(self.h * self.w))
        # num = 10
        init_cells = set()
        while len(init_cells) < num:
            i = random.randrange(self.h)
            j = random.randrange(self.w)
            if (i, j) not in self.mine_pos and (i, j) not in init_cells:
                init_cells.add((i, j))
        
        return init_cells

    def print_board(self):
        os.system('cls')
        for i in range(self.h):
            for j in range(self.w):
                if self.shown_cell[i][j]:
                    print(self.board[i][j], end=' ')
                else:
                    print('?', end=' ')
            print()

class Literal:
    def __init__(self, cell, is_posi):
        self.cell = cell
        self.posi = is_posi

    def __eq__(self, other):
        return self.cell == other.cell and self.posi == other.posi
    
    def __str__(self):
        return str(self.cell) + ('' if self.posi else "'")
    
    def __hash__(self):
        return hash(str(self))
    
class Clause:
    def __init__(self, literals=[]):
        self.literals = set(literals)
        
    
    def __str__(self):
        return "[" + ' '.join([str(l) for l in self.literals]) + "]"
    
    def __eq__(self, other):
        return self.literals == other.literals
    
    def __len__(self):
        return len(self.literals)
    
    def __hash__(self):
        return hash(str(self))
    
    def __copy__(self):
        return Clause(self.literals.copy())
    
    
class KB:
    def __init__(self, clauses=set()):
        self.clauses = clauses

    def insert(self, clause: Clause, KB0):
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
    if len(a) > 2 and len(b) > 2:
        return a, b
    
    if a == b:
        return a, None
    
    if a.literals.issubset(b.literals):
        return a, None
    
    if b.literals.issubset(a.literals):
        return b, None
    
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
    def __init__(self, game: Game):
        self.game = game
        self.KB = KB(set())
        self.KB0 = KB(set())
        self.mine = set()
        self.safe = set()
        for i in self.game.get_init_safe_cells():
            self.safe.add(i)
            self.KB.insert(Clause([Literal(i, False)]), self.KB0)

    def play(self):
        unmarked_cnt = 0
        while unmarked_cnt <= 10:
            self.game.print_board()
            print(f"# in KB: {len(self.KB.clauses)}, # in KB0: {len(self.KB0.clauses)}")
            # for clause in self.KB.clauses:
            #     print(clause, len(clause))
            # print("----")
            # for clause in self.KB0.clauses:
            #     print(clause)
            print(f"# single clause in KB: {len([clause for clause in self.KB.clauses if len(clause) == 1])}")
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
                    print(f"Open cell {lit.cell} with {lit.posi}")
                    if lit.posi:
                        if self.game.open_cell(lit.cell, False) == -1:
                            print('Game Over!')
                            exit(0)
                        self.mine.add(lit.cell)
                    else:
                        if self.game.open_cell(lit.cell, True) == -1:
                            print('Game Over!')
                            exit(0)
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
            KB_clause = list(self.KB.clauses.copy())
            print("entering pairwise matching")
            unmarked_cnt += 1
            for idx, i in tqdm(enumerate(KB_clause)):
                for j in KB_clause[idx+1:]:
                    if i in self.KB.clauses:
                        self.KB.clauses.remove(i)
                    if j in self.KB.clauses:
                        self.KB.clauses.remove(j)
                    if(len(i) == 0 or len(j) == 0):
                        continue
                    a, b = matching_clauses(Clause(i.literals.copy()), Clause(j.literals.copy()))
                    
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
                else:
                    print("Win!")
                exit(0)
        print("Stuck")
        exit(0)
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
    args = parser.parse_args()
    game = Game(args.difficulty)
    player = Player(game)
    player.play()
