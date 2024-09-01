from random import choice, randint, uniform, choices
from multiprocessing import Pool, cpu_count
from base64 import b64encode
from json import dumps

CLOSED_CELL = 2
OPEN_CELL = 4
BOT_CELL = 8

ADJ_CELLS = [(1, 0), (0, 1), (0, -1), (-1, 0)]
# MOVE_CELLS = [(1, 0), (0, 1), (0, -1), (-1, 0)]
MOVE_CELLS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
GRID_SIZE = 5
CONVERGENCE_LIMIT = 1e-4
MAX_ITERS = 1000

# Common Methods
# parts of this code is from my class written for the group project 1,2,3
def get_neighbors(cell, grid, filter, is_special = False):
    neighbors = []

    for move in ADJ_CELLS:
        x_cord = cell[0] + move[0]
        y_cord = cell[1] + move[1]
        if ( is_special or (
            (0 <= x_cord < GRID_SIZE)
            and (0 <= y_cord < GRID_SIZE)
            and (grid[x_cord][y_cord].cell_type & filter))
        ):
            neighbors.append((x_cord, y_cord))

    return neighbors

# parts of this code is from my class written for the group project 1,2,3
class Cell:
    def __init__(self, i, j, cell_type):
        self.pos = (i, j)
        self.cell_type = cell_type
        self.original_states = 0
        self.bot_time_step = float(GRID_SIZE**2)

# parts of this code is from my class written for the group project 1,2,3
class Ship:
    def __init__(self):
        self.size = GRID_SIZE
        self.grid = [[Cell(i, j, CLOSED_CELL) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)] # grid is a list of list with each cell as a class
        self.open_cells = []
        self.bot = (0, 0)
        self.corner_cell = (0, 0)
        self.init_plots = False
        self.print_data = [[0 for i in range(self.size)] for j in range(self.size)]

        self.generate_grid()

    # parts of this code is from my work on group project 1,2,3
    def get_cell(self, cord):
        return self.grid[cord[0]][cord[1]]

    # parts of this code is from my work on group project 1,2,3
    def generate_grid(self):
        self.assign_start_cell()
        self.unblock_closed_cells()
        self.unblock_dead_ends()
        self.compute_adj_cells()
        self.place_players()

    # parts of this code is from my work on group project 1,2,3
    def compute_adj_cells(self):
        for cell_cord in self.open_cells:
            cell = self.get_cell(cell_cord)
            neighbors = get_neighbors(
                cell_cord,
                self.grid,
                OPEN_CELL
            )

            cell.adj_cells = neighbors

    # parts of this code is from my work on group project 1,2,3
    def assign_start_cell(self):
        random_row = randint(0, self.size - 1)
        random_col = randint(0, self.size - 1)
        self.grid[random_row][random_col].cell_type = OPEN_CELL
        self.grid[random_row][random_col].original_states = 1
        self.open_cells.append((random_row, random_col))

    # parts of this code is from my work on group project 1,2,3
    def unblock_closed_cells(self):
        available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)
        while len(available_cells):
            closed_cell = choice(available_cells)
            self.get_cell(closed_cell).cell_type = OPEN_CELL
            self.get_cell(closed_cell).original_states = 1
            self.open_cells.append(closed_cell)
            available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)

    # parts of this code is from my work on group project 1,2,3
    def unblock_dead_ends(self):
        dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
        half_len = len(dead_end_cells)/2

        while half_len > 0:
            dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
            half_len -= 1
            if len(dead_end_cells):
                continue
            dead_end_cell = choice(dead_end_cells)
            closed_neighbors = get_neighbors(dead_end_cell, self.grid, CLOSED_CELL)
            random_cell = choice(closed_neighbors)
            self.get_cell(random_cell).cell_type = OPEN_CELL
            self.get_cell(random_cell).original_states = 1
            self.open_cells.append(random_cell)

    # parts of this code is from my work on group project 1,2,3
    def cells_with_one_open_neighbor(self, cell_type):
        results = []
        for row in range(self.size):
            for col in range(self.size):
                if ((self.grid[row][col].cell_type & cell_type) and
                    len(get_neighbors(
                        (row, col), self.grid, OPEN_CELL
                    )) == 1):
                    results.append((row, col))
        return results

    # parts of this code is from my work on group project 1,2,3
    def place_players(self):
        self.bot = choice(self.open_cells)
        self.get_cell(self.bot).cell_type = BOT_CELL

    # parts of this code is from my work on group project 1,2,3
    def print_grid(self, time = 2.0):
        for i in range(self.size):
            for j in range(self.size):
                self.print_data[i][j] = self.grid[i][j].original_states

        from matplotlib import pyplot
        if not self.init_plots:
            self.fig, self.ax = pyplot.subplots()
            self.image = pyplot.imshow(self.print_data, cmap='tab20')
            self.init_plots = True

        self.image.set_data(self.print_data)
        self.fig.canvas.draw_idle()
        pyplot.pause(time)

    def pred_best_moves_dfs(self):
        self.dfs_moves = []
        iter = 0
        final_moves = {}
        for cells in self.open_cells:
            final_moves[cells] = 1

        new_moves = {}
        while(True):
            iter += 1
            total_count = 0
            new_moves = {}
            best_move = ((0, 0), len(self.open_cells))
            count_best = 0
            for move in MOVE_CELLS:
                new_moves[move] = {}
                count = 0
                for open_cell in final_moves:
                    new_cord = (open_cell[0] + move[0], open_cell[1] + move[1])
                    if new_cord[0] < 0 or new_cord[0] >= GRID_SIZE or new_cord[1] < 0 or new_cord[1] >= GRID_SIZE or self.get_cell(new_cord).cell_type == CLOSED_CELL:
                        new_moves[move][open_cell] = 1
                        continue

                    new_moves[move][new_cord] = 1

                if len(new_moves[move]) == best_move[1]:
                    count_best += 1

                if len(new_moves[move]) < best_move[1]:
                    best_move = (move, len(new_moves[move]))
                    count_best = 1

            if count_best > len(MOVE_CELLS)/2:
                best_move = (choice(MOVE_CELLS), count_best)

            self.dfs_moves.append(best_move[0])
            if best_move[1] == 1:
                for key in new_moves[best_move[0]]:
                    self.dfs_pred_pos = key
                    break
                return

            for move in MOVE_CELLS:
                final_moves.clear()

            final_moves = new_moves[best_move[0]]

        return

    def encode(self, obj):
        return b64encode(dumps(obj).encode()).decode()

    def pred_best_moves_bfs(self):
        self.bfs_moves = []
        iter = 0
        all_moves = {}
        for cells in self.open_cells:
            all_moves[cells] = 1

        new_moves = {}
        all_possible_moves = [(all_moves, [(0, 0)])]
        explored = []
        while(all_possible_moves):
            new_moves = {}
            if iter % GRID_SIZE == 0:
                all_possible_moves = sorted(all_possible_moves, key = lambda data:len(data[1]))

            curr_move_dict, best_path = all_possible_moves.pop()
            # print(curr_move_dict, best_path)
            count_best =  len(self.open_cells)
            for move in MOVE_CELLS:
                iter += 1
                new_moves = {}
                new_cell = 0
                for open_cell in curr_move_dict:
                    new_cord = (open_cell[0] + move[0], open_cell[1] + move[1])
                    if new_cord[0] < 0 or new_cord[0] >= GRID_SIZE or new_cord[1] < 0 or new_cord[1] >= GRID_SIZE or self.get_cell(new_cord).cell_type == CLOSED_CELL:
                        new_moves[open_cell] = 1
                        continue

                    new_moves[new_cord] = 1
                    new_cell = 1

                if new_cell == 0:
                    continue


                if len(new_moves) == 1:
                    best_path.pop(0)
                    self.bfs_moves = best_path + [move]
                    # print(self.bfs_moves)
                    for key in new_moves:
                        self.bfs_pred_pos = key
                        break
                    return

                result = [key[0]*GRID_SIZE+key[1] for key in new_moves]
                result.sort() # sort might help us
                encoded_val = self.encode(result)
                if encoded_val not in explored:
                    explored.append(encoded_val)
                    all_possible_moves.append((new_moves, best_path + [move]))

        return []

class Bot:
    def __init__(self, ship):
        self.ship = ship
        self.bot_pos = ship.bot

    def move_bot(self, move):
        new_cord = (self.bot_pos[0] + move[0], self.bot_pos[1] + move[1])
        if new_cord[0] < 0 or new_cord[0] >= GRID_SIZE or new_cord[1] < 0 or new_cord[1] >= GRID_SIZE or self.ship.get_cell(new_cord).cell_type == CLOSED_CELL:
            return

        self.ship.get_cell(self.bot_pos).cell_type = OPEN_CELL
        self.ship.get_cell(new_cord).cell_type = BOT_CELL
        self.bot_pos = new_cord

    def check_with_dfs(self):
        for move in self.ship.dfs_moves:
            self.move_bot(move)

        if self.ship.dfs_pred_pos == self.bot_pos:
            return True
        else:
            return False

    def check_with_bfs(self):
        for move in self.ship.bfs_moves:
            self.move_bot(move)

        if self.ship.bfs_pred_pos == self.bot_pos:
            return True
        else:
            return False

class Data_Analysis:
    def __init__(self):
        self.dfs_w = self.bfs_w = 0
        self.dfs_moves = self.bfs_moves = 0

    def update_dfs(self, moves, incr = 1):
        self.dfs_w += incr
        self.dfs_moves += moves

    def update_bfs(self, moves, incr = 1):
        self.bfs_w += incr
        self.bfs_moves += moves

# parts of this code is from my work on group project 1,2,3
def run_sim(iters):
    data = Data_Analysis()
    for i in range(iters[0]):
        if iters[1]:
            print(i, end='\r')
        ship = Ship()
        ship.pred_best_moves_dfs()
        ship.pred_best_moves_bfs()
        bot = Bot(ship)
        if(bot.check_with_dfs()):
            data.update_dfs(len(ship.dfs_moves))

        if(bot.check_with_bfs()):
            data.update_bfs(len(ship.bfs_moves))

        del bot
        del ship

    if iters[1]:
        print()
        print(f"Using DFS, we predicted {data.dfs_w/iters*100}%, and it took {data.dfs_moves/iters} moves on an avg.")
        print(f"Using BFS, we predicted {data.bfs_w/iters*100}%, and it took {data.bfs_moves/iters} moves on an avg.")

    return data

# parts of this code is from my work on group project 1,2,3
def multi_sim():
    core_count = cpu_count()
    arg_data = [[int(MAX_ITERS/core_count), False] for i in range(core_count)]
    max_iters = int(MAX_ITERS/core_count)*core_count
    final_data = Data_Analysis()
    with Pool(processes=core_count) as p:
        for data in p.map(run_sim, arg_data):
            final_data.update_dfs(data.dfs_moves, data.dfs_w)
            final_data.update_bfs(data.bfs_moves, data.bfs_w)

        print(f"Bot has the following moves {MOVE_CELLS} on a {GRID_SIZE}*{GRID_SIZE} grid")
        print(f"Following data has been run {max_iters} times")
        print(f"Using DFS, we predicted {final_data.dfs_w/max_iters*100}%, and it took {final_data.dfs_moves/max_iters} moves on an avg.")
        print(f"Using BFS, we predicted {final_data.bfs_w/max_iters*100}%, and it took {final_data.bfs_moves/max_iters} moves on an avg.")


if __name__ == "__main__":
    # run_sim([MAX_ITERS, True])
    multi_sim()
