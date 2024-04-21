from random import choice, random
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from time import time

CLOSED_CELL = 0
TELEPORT_CELL = 1
OPEN_CELL = 2
CREW_CELL = 4
BOT_CELL = 8
TOTAL_ITERATIONS = 10000 # iterations for same ship layout and different bot/crew positions
TOTAL_BOTS = 2
GRID_SIZE = 11
SUCCESS = 1
FAILURE = 0
CONV_ITERATIONS_LIMIT = 1000

#Debugging
RAND_CLOSED_CELLS = True
VISUALIZE = False
MAX_CORES = cpu_count()

ALL_CREW_MOVES = [(1, 0), (0, 1), (-1, 0), (0, -1)]
ALL_BOT_MOVES = [(1, 0), (0, 1), (1, 1), (-1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1)]

def get_manhattan_distance(cell_1, cell_2):
    return abs(cell_1[0] - cell_2[0]) + abs(cell_1[1] - cell_2[1])

class cell:
    def __init__(self):
        self.state = OPEN_CELL
        self.is_move = 1
        self.bot_moves = 0.0
        self.no_bot_moves = 0.0
        self.bot_distance = 0

class SHIP:
    def __init__(self):
        self.size = GRID_SIZE
        self.grid = [[cell() for j in range(self.size)] for i in range(self.size)]
        self.open_cells = [ (j, i) for j in range(self.size) for i in range(self.size)]
        self.moves_lookup = {}
        self.crew_pos = (0, 0)
        self.bot_pos = (0, 0)
        self.convergence_iters_limit = 0
        self.set_grid()
        self.place_players()

    def set_grid(self):
        mid_point = (int)((self.size-1)/2)
        self.teleport_cell = (mid_point, mid_point)
        self.set_state(self.teleport_cell, TELEPORT_CELL)
        self.grid[mid_point][mid_point].no_bot_moves = 0
        other_points = [mid_point - 1, mid_point + 1]
        for i in other_points:
            for j in other_points:
                self.open_cells.remove((i, j))
                self.grid[i][j].no_bot_moves = 0
                self.set_state((i, j), CLOSED_CELL)

        if RAND_CLOSED_CELLS:
            self.place_random_closed_cells()

    def place_random_closed_cells(self):
        random_closed = 10
        ignore_cells = [self.teleport_cell]
        for i in range(-2, 3):
            if i == 0:
                continue

            for (row,col) in [(0, 1*i),(1*i, 0)]:
                x_cord = self.teleport_cell[0] + row
                y_cord = self.teleport_cell[1] + col
                ignore_cells.append((x_cord, y_cord))

        while(True):
            random_cell = choice(self.open_cells)
            if random_cell not in ignore_cells:
                random_closed -= 1
                self.set_state(random_cell, CLOSED_CELL)
                self.open_cells.remove(random_cell)

            if not random_closed:
                break

    def place_players(self):
        while(True):
            self.crew_pos = choice(self.open_cells)
            if len(self.get_all_moves(self.crew_pos, ~CLOSED_CELL)) != 0:
                break

        crew_state = TELEPORT_CELL | CREW_CELL if self.get_state(self.crew_pos) & TELEPORT_CELL else CREW_CELL
        self.set_state(self.crew_pos, crew_state)
        # self.set_moves(self.crew_pos, 1)
        self.open_cells.remove(self.crew_pos)
        while(True):
            self.bot_pos = choice(self.open_cells)
            if len(self.get_all_moves(self.bot_pos, ~CLOSED_CELL, False)) != 0:
                break

        bot_state = TELEPORT_CELL | BOT_CELL if self.get_state(self.bot_pos) & TELEPORT_CELL else BOT_CELL
        self.set_state(self.bot_pos, bot_state)
        self.open_cells.remove(self.bot_pos)

    def set_state(self, pos, state_val):
        self.grid[pos[0]][pos[1]].state = state_val

    def set_moves(self, pos, moves):
        self.grid[pos[0]][pos[1]].bot_moves = moves
        self.grid[pos[0]][pos[1]].no_bot_moves = moves

    def get_state(self, pos):
        return self.grid[pos[0]][pos[1]].state

    def get_cell(self, pos):
        return self.grid[pos[0]][pos[1]]

    def print_ship(self):
        for i in range(self.size):
            for j in range(self.size):
                # print(("%20s " %self.grid[i][j].no_bot_moves), end=" ")
                print(f"{self.grid[i][j].state}", end=" ")
            print()
        print("len ::", len(self.open_cells))

    def reset_grid(self):
        for cell in self.open_cells:
            self.set_state(cell, OPEN_CELL)

        self.set_state(self.teleport_cell, TELEPORT_CELL)
        crew_state = TELEPORT_CELL | CREW_CELL if self.get_state(self.crew_pos) & TELEPORT_CELL else CREW_CELL
        self.set_state(self.crew_pos, crew_state)
        bot_state = TELEPORT_CELL | BOT_CELL if self.get_state(self.bot_pos) & TELEPORT_CELL else BOT_CELL
        self.set_state(self.bot_pos, bot_state)

    def reset_positions(self):
        for cell in self.open_cells:
            self.set_state(cell, OPEN_CELL)

        self.set_state(self.crew_pos, OPEN_CELL)
        self.set_state(self.bot_pos, OPEN_CELL)
        self.set_state(self.teleport_cell, TELEPORT_CELL)
        self.open_cells.append(self.bot_pos)
        self.open_cells.append(self.crew_pos)
        self.place_players()

    def get_all_moves(self, curr_pos, filter = OPEN_CELL | TELEPORT_CELL, is_crew = True):
        neighbors = []

        all_moves = ALL_CREW_MOVES if is_crew else ALL_BOT_MOVES
        for (row,col) in all_moves:
            x_cord = curr_pos[0] + row
            y_cord = curr_pos[1] + col
            if (
                (0 <= x_cord < self.size)
                and (0 <= y_cord < self.size)
                and (self.get_state((x_cord, y_cord)) & filter)
            ):
                neighbors.append((x_cord, y_cord))

        return neighbors

    def calc_no_bot_steps(self):
        total_iters = 0
        while(True):
            total_range = self.size**2
            for i in range(self.size):
                for j in range(self.size):
                    curr_cell = self.get_cell((i, j))
                    if curr_cell.state != CLOSED_CELL and (i, j) != self.teleport_cell:
                        neighbors = self.get_all_moves((i, j), ~(CLOSED_CELL))
                        moves_len = len(neighbors)
                        old_sum = curr_cell.no_bot_moves
                        if moves_len:
                            curr_cell.no_bot_moves = 1 + sum(self.get_cell(cell).no_bot_moves for cell in neighbors)/moves_len
                        else:
                            curr_cell.no_bot_moves = 0

                        if old_sum == curr_cell.no_bot_moves:
                            total_range -= 1
                    else:
                        total_range -= 1

            total_iters += 1
            if total_range == 0 or total_iters >= CONV_ITERATIONS_LIMIT:
                self.convergence_iters_limit = total_iters
                break

    def calc_bot_steps(self):
        moves_list = []
        time_lookup = {}

        for i in range(self.size):
            inner_list = []
            for j in range(self.size):
                inner_list.append(self.get_cell((i, j)).no_bot_moves)
            moves_list.append(inner_list)

        for row in range(self.size):
            for col in range(self.size):
                total_iters = 0
                bot_pos = (row, col)
                bot_cell = self.get_cell(bot_pos)
                moves_copy = [list(inner_list) for inner_list in moves_list]
                moves_copy[row][col] = 0
                if bot_cell.state == CLOSED_CELL:
                    continue

                bot_adj_cells = self.get_all_moves(bot_pos, ~CLOSED_CELL)
                for neighbor in bot_adj_cells:
                    self.get_cell(neighbor).bot_distance = 1
                    espace_cells = self.get_all_moves(neighbor, ~CLOSED_CELL)
                    max_distance = 0
                    for escape in espace_cells:
                        bot_distance = get_manhattan_distance(neighbor, escape)
                        self.get_cell(escape).bot_distance = bot_distance
                        if max_distance < bot_distance:
                            max_distance = bot_distance

                    for escape in espace_cells:
                        if max_distance != self.get_cell(escape).bot_distance or escape == bot_pos:
                            self.get_cell(escape).bot_distance = -1

                while(True):
                    total_range = self.size**2
                    for i in range(self.size):
                        for j in range(self.size):
                            curr_cell = self.get_cell((i, j))
                            if curr_cell.state != CLOSED_CELL and (i, j) != self.teleport_cell and (i, j) != bot_pos:
                                neighbors = self.get_all_moves((i, j), ~CLOSED_CELL)
                                curr_bot_distance = self.get_cell((i, j)).bot_distance
                                if curr_bot_distance == 1:
                                    for neighbor in list(neighbors):
                                        if self.get_cell(neighbor).bot_distance == -1:
                                            neighbors.remove(neighbor)

                                moves_len = len(neighbors)
                                old_sum = moves_copy[i][j]
                                if moves_len:
                                    moves_copy[i][j] = 1 + (sum(moves_copy[cell[0]][cell[1]] for cell in neighbors)/moves_len)
                                else:
                                    moves_copy[i][j] += 1

                                if old_sum == moves_copy[i][j]:
                                    total_range -= 1
                            else:
                                total_range -= 1

                    total_iters += 1
                    if total_range == 0 or total_iters >= self.convergence_iters_limit:
                        break

                time_lookup[bot_pos] = moves_copy
                for i in range(self.size):
                    for j in range(self.size):
                        curr_cell = self.get_cell((i, j))
                        curr_cell.bot_distance = 0

        self.calc_movement_lookup(time_lookup)

    def calc_movement_lookup(self, time_lookup):
        self.moves_lookup.clear()
        state_dict = {}

        for iters in range(self.convergence_iters_limit):
            for bot_pos in time_lookup:
                if self.get_state(bot_pos) & CLOSED_CELL:
                    continue

                # bot_len_pos = bot_pos[0] + bot_pos[1]*self.size
                if bot_pos not in self.moves_lookup:
                    self.moves_lookup[bot_pos] = {}

                if bot_pos not in state_dict:
                    state_dict[bot_pos] = {}

                curr_bot_dict = self.moves_lookup[bot_pos]
                curr_bot_state = state_dict[bot_pos]
                curr_bot_moves = self.get_all_moves(bot_pos, ~CLOSED_CELL, False)
                curr_bot_moves.append(bot_pos)
                for i in range(self.size):
                    for j in range(self.size):
                        crew_pos = (i, j)
                        if self.get_state(crew_pos) & CLOSED_CELL or crew_pos == bot_pos:
                            continue

                        if crew_pos not in curr_bot_dict:
                            curr_bot_dict[crew_pos] = {}

                        if crew_pos not in curr_bot_state:
                            curr_bot_state[crew_pos] = 0.0

                        curr_action_dict = curr_bot_dict[crew_pos]
                        for bot_action in curr_bot_moves:
                            if bot_action == crew_pos:
                                continue

                            if bot_action not in curr_action_dict:
                                curr_action_dict[bot_action] = 0.0

                            if bot_action not in state_dict:
                                state_dict[bot_action] = {}

                            curr_action_dict[bot_action] = 0.0
                            if crew_pos == self.teleport_cell:
                                continue

                            action_state = state_dict[bot_action]
                            curr_crew_moves = self.get_all_moves(crew_pos, ~CLOSED_CELL)
                            curr_bot_distance = get_manhattan_distance(crew_pos, bot_action)
                            if curr_bot_distance == 1:
                                possible_moves = []
                                max_distance = 0
                                # print(curr_crew_moves, bot_action, crew_pos)
                                for crew_move in curr_crew_moves:
                                    if crew_move == bot_action:
                                        continue

                                    distance = get_manhattan_distance(crew_move, bot_action)
                                    possible_moves.append((crew_move, distance))
                                    if max_distance < distance:
                                        max_distance = distance

                                curr_crew_moves.clear()
                                # print(possible_moves, max_distance, bot_action, crew_pos)
                                for move_vs_manhattan in possible_moves:
                                    if max_distance == move_vs_manhattan[1]:
                                        curr_crew_moves.append(move_vs_manhattan[0])

                            if not curr_crew_moves:
                                curr_crew_moves.append(crew_pos)

                            move_prob = 1/len(curr_crew_moves)
                            for move in curr_crew_moves:
                                if move not in action_state:
                                    action_state[move] = 0.0

                                crew_reward = -1*time_lookup[bot_action][move[0]][move[1]]
                                curr_action_dict[bot_action] += (move_prob*(crew_reward + action_state[move]))

                        curr_bot_state[crew_pos] = -1 + curr_action_dict[max(curr_action_dict, key=lambda bot_action:curr_action_dict[bot_action])]


class parent_bot:
    def __init__(self, ship):
        self.ship = ship
        self.local_crew_pos = self.ship.crew_pos
        self.local_bot_pos = self.ship.bot_pos

    def move_bot(self):
        return

    def move_crew(self):
        return bool(True)

    def visualize_grid(self):
        if not VISUALIZE:
            return

        data = []
        for i in range(self.ship.size):
            inner_list = []
            for j in range(self.ship.size):
                inner_list.append(self.ship.get_state((i, j)))

            data.append(inner_list)
        from matplotlib import pyplot
        fig, ax = pyplot.subplots()
        ax.matshow(data, cmap='seismic')
        pyplot.show()

    def start_rescue(self):
        total_iter = 0
        if self.ship.get_state(self.local_crew_pos) & TELEPORT_CELL:
            return total_iter, SUCCESS

        while(True):
            self.visualize_grid()
            total_iter += 1
            self.move_bot()
            if self.move_crew():
                self.visualize_grid()
                return total_iter, SUCCESS

            if total_iter > 999:
                self.visualize_grid()
                return total_iter, FAILURE

class no_bot(parent_bot):
    def __init__(self, ship):
        super(no_bot, self).__init__(ship)
        if self.ship.get_state(self.local_bot_pos) & TELEPORT_CELL:
            self.ship.set_state(self.local_bot_pos, TELEPORT_CELL)
        else:
            self.ship.set_state(self.local_bot_pos, OPEN_CELL)

    def move_crew(self):
        neighbors = self.ship.get_all_moves(self.local_crew_pos)
        if not neighbors:
            return False

        next_cell = choice(neighbors)
        self.ship.set_state(self.local_crew_pos, OPEN_CELL)
        old_pos = self.local_crew_pos
        self.local_crew_pos = next_cell
        next_state = CREW_CELL
        if self.ship.get_state(next_cell) & TELEPORT_CELL:
            next_state |= TELEPORT_CELL
            self.ship.set_state(next_cell, next_state)
            return True

        self.ship.set_state(next_cell, next_state)
        return False

class bot(parent_bot):
    def __init__(self, ship):
        super(bot, self).__init__(ship)

    def move_bot(self):
        bot_movements = self.ship.get_all_moves(self.local_bot_pos, OPEN_CELL | TELEPORT_CELL, False)
        bot_movements.append(self.local_bot_pos)
        best_move = max(bot_movements, key = lambda move:self.ship.moves_lookup[self.local_bot_pos][self.local_crew_pos][move])
        if not best_move:
            return

        old_pos = self.local_bot_pos
        old_state = TELEPORT_CELL if self.ship.get_state(old_pos) & TELEPORT_CELL else OPEN_CELL
        self.ship.set_state(self.local_bot_pos, old_state)
        self.local_bot_pos = best_move
        next_state = BOT_CELL
        if self.ship.get_state(best_move) & TELEPORT_CELL:
            next_state |= TELEPORT_CELL

        self.ship.set_state(best_move, next_state)

    def move_crew(self):
        neighbors = self.ship.get_all_moves(self.local_crew_pos)
        if self.local_bot_pos in neighbors:
            max_distance = 0
            new_neighbors = []
            for neighbor in neighbors:
                if neighbor != self.local_bot_pos:
                    bot_distance = get_manhattan_distance(neighbor, self.local_bot_pos)
                    new_neighbors.append((neighbor, bot_distance))
                    if max_distance < bot_distance:
                        max_distance = bot_distance

            neighbors.clear()
            for neighbor in new_neighbors:
                if max_distance == neighbor[1]:
                    neighbors.append(neighbor[0])

        if not neighbors:
            return False

        next_cell = choice(neighbors)

        self.ship.set_state(self.local_crew_pos, OPEN_CELL)
        old_pos = self.local_crew_pos
        self.local_crew_pos = next_cell
        next_state = CREW_CELL
        if self.ship.get_state(next_cell) & TELEPORT_CELL:
            next_state |= TELEPORT_CELL
            self.ship.set_state(next_cell, next_state)
            return True

        self.ship.set_state(next_cell, next_state)
        return False

class DETAILS:
    def __init__(self):
        self.success = self.failure = 0.0
        self.s_moves = self.f_moves = 0.0
        self.max_success = self.min_success = 0
        self.distance = 0.0
        self.dest_dist = 0.0

    def update_min_max(self, moves):
        if self.max_success < moves:
            self.max_success = moves

        if self.min_success > moves:
            self.min_success = moves

    def update(self, new_detail):
        self.s_moves += new_detail.s_moves
        self.success += new_detail.success
        self.f_moves += new_detail.f_moves
        self.failure += new_detail.failure
        self.distance += new_detail.distance
        self.dest_dist += new_detail.dest_dist
        self.update_min_max(new_detail.max_success)
        self.update_min_max(new_detail.min_success)

    def get_avg(self, total_itr):
        if self.success:
            self.s_moves /= self.success

        if self.failure:
            self.f_moves /= self.failure

        self.success /= total_itr
        self.failure /= total_itr
        self.distance /= total_itr
        self.dest_dist /= total_itr

def bot_fac(itr, myship):
    if itr % TOTAL_BOTS  == 0:
        return no_bot(myship)
    else:
        return bot(myship)

def run_sim(sim_range):
    ship = SHIP()
    ship.calc_no_bot_steps()
    ship.calc_bot_steps()
    avg_moves = [DETAILS() for itr in range(TOTAL_BOTS)]
    for _ in sim_range:
        # print(_, end = "\r")
        dest_dist = get_manhattan_distance(ship.crew_pos, ship.teleport_cell)
        for itr in range(TOTAL_BOTS):
            test_bot = bot_fac(itr, ship)
            moves, result = test_bot.start_rescue()
            ship.reset_grid()
            if result:
                avg_moves[itr].update_min_max(moves)
                avg_moves[itr].s_moves += moves
                avg_moves[itr].success += 1
            else:
                avg_moves[itr].f_moves += moves
                avg_moves[itr].failure += 1

            distance = 0 if test_bot.__class__ is no_bot else get_manhattan_distance(ship.bot_pos, ship.crew_pos)
            avg_moves[itr].distance += distance
            avg_moves[itr].dest_dist += dest_dist
            del test_bot

        ship.reset_positions()

    # print()
    del ship
    return avg_moves

def print_header(total_itr = TOTAL_ITERATIONS):
    print("Total iterations performed for layout is", total_itr)
    print("%3s %18s %18s %18s %18s %18s %18s %18s %18s" % ("No", "Avg Suc Moves", "Success Rate", "Min Suc. Moves", "Max Suc. Moves", "Avg Fail Moves", "Failure Rate", "Avg Bot Crew Dist", "Crew Teleport Dist"))

def print_data(final_data, itr, total_itr = TOTAL_ITERATIONS):
    final_data[itr].get_avg(total_itr)
    print(("%3s %18s %18s %18s %18s %18s %18s %18s %18s" % (itr, final_data[itr].s_moves, final_data[itr].success, final_data[itr].min_success, final_data[itr].max_success, final_data[itr].f_moves, final_data[itr].failure, final_data[itr].distance, final_data[itr].dest_dist)))

def run_multi_sim():
    core_count = MAX_CORES
    arg_data = [range(0, TOTAL_ITERATIONS) for i in range(core_count)]
    avg_moves = [[DETAILS() for itr in range(TOTAL_BOTS)] for _ in range(core_count)]
    with Pool(processes=core_count) as p:
        for layout, final_data in enumerate(p.map(run_sim, arg_data)):
            curr_ship = avg_moves[layout]
            for bot_no, data in enumerate(final_data):
                curr_ship[bot_no].update(data)

        print_header()
        for layout in range(core_count):
            print("Layout no. :: ", layout)
            curr_ship = avg_moves[layout]
            for itr in range(TOTAL_BOTS):
                print_data(curr_ship, itr)

def single_sim(total_itr):
    final_data = run_sim(range(0, total_itr))

    print_header(total_itr)
    for itr in range(TOTAL_BOTS):
        print_data(final_data, itr, total_itr)

def single_run():
    ship = SHIP()
    ship.calc_no_bot_steps()
    ship.calc_bot_steps()
    ship.print_ship()
    for itr in range(TOTAL_BOTS):
        test_bot = bot_fac(itr, ship)
        print(test_bot.start_rescue())
        ship.reset_grid()


if __name__ == '__main__':
    begin = time()
    # single_run()
    # single_sim(1000)
    run_multi_sim()
    end = time()
    print(end-begin)
