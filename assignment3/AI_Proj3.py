from random import choice, random
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from time import time
import csv
import os

# cell constants
CLOSED_CELL = 0
TELEPORT_CELL = 1
OPEN_CELL = 2
CREW_CELL = 4
BOT_CELL = 8

# layout constantss
GRID_SIZE = 11
SUCCESS = 1
FAILURE = 0
CONV_ITERATIONS_LIMIT = 1000
CONVERGENCE_LIMIT = 1e-5

#Debugging
RAND_CLOSED_CELLS = 10
TOTAL_ITERATIONS = 10000 # iterations for same ship layout and different bot/crew positions
TOTAL_CONFIGS = 2
MAX_CORES = cpu_count()
VISUALIZE = False

# moves constants
ALL_CREW_MOVES = [(1, 0), (0, 1), (-1, 0), (0, -1)]
ALL_BOT_MOVES = [(1, 0), (0, 1), (1, 1), (-1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1)]

def get_manhattan_distance(cell_1, cell_2):
    return abs(cell_1[0] - cell_2[0]) + abs(cell_1[1] - cell_2[1])

class CELL:
    def __init__(self, i, j):
        self.state = OPEN_CELL
        self.no_bot_moves = float(GRID_SIZE**2)
        pos = (i, j)
        state = j + i*GRID_SIZE

class SHIP:
    def __init__(self):
        self.size = GRID_SIZE
        self.grid = [[CELL(i, j) for j in range(self.size)] for i in range(self.size)]
        self.open_cells = [ (j, i) for j in range(self.size) for i in range(self.size)]
        self.crew_pos = (0, 0)
        self.bot_pos = (0, 0)
        self.ideal_iters_limit = 0
        self.global_min_max = -(11**4*9*4)
        self.closed_cells =[]
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
        random_closed = RAND_CLOSED_CELLS
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
                self.closed_cells.append(random_cell)
                self.open_cells.remove(random_cell)

            if not random_closed:
                break

    def place_players(self):
        while(True):
            self.crew_pos = choice(self.open_cells)
            if self.search_path():
                break

        crew_state = TELEPORT_CELL | CREW_CELL if self.get_state(self.crew_pos) & TELEPORT_CELL else CREW_CELL
        self.set_state(self.crew_pos, crew_state)
        self.open_cells.remove(self.crew_pos)

        while(True):
            self.bot_pos = choice(self.open_cells)
            if self.search_path(False):
                break

        bot_state = TELEPORT_CELL | BOT_CELL if self.get_state(self.bot_pos) & TELEPORT_CELL else BOT_CELL
        self.set_state(self.bot_pos, bot_state)
        self.open_cells.remove(self.bot_pos)

    def set_state(self, pos, state_val):
        self.grid[pos[0]][pos[1]].state = state_val

    def set_moves(self, pos, moves):
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

    def search_path(self, is_crew = True):
        curr_pos = self.crew_pos if is_crew else self.bot_pos

        bfs_queue = []
        visited_cells = set()
        bfs_queue.append((curr_pos, [curr_pos]))

        while bfs_queue:
            current_cell, path_traversed = bfs_queue.pop(0)
            if current_cell == self.teleport_cell:
                path_traversed.pop(0)
                return path_traversed
            elif (current_cell in visited_cells):
                continue

            visited_cells.add(current_cell)
            neighbors = self.get_all_moves(current_cell, ~CLOSED_CELL, is_crew)
            for neighbor in neighbors:
                if (neighbor not in visited_cells):
                    bfs_queue.append((neighbor, path_traversed + [neighbor]))

        return [] #God forbid, this should never happen

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

    def perform_initial_calcs(self):
        # no bot
        self.calc_no_bot_steps()

        # bot - always calc no bot first, we can reuse those time steps instead of starting afresh
        self.set_calcs_lookup()
        self.set_state_details()
        self.perform_moves_iteration()
        self.perform_value_iteration()
        self.unset_global_states()

    def unset_global_states(self):
        del self.indi_states_lookup
        del self.time_lookup
        del self.crew_moves
        del self.bot_moves
        del self.manhattan_lookup
        del self.bot_vs_crew_state

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

                        if curr_cell.no_bot_moves - old_sum < CONVERGENCE_LIMIT:
                            total_range -= 1
                    else:
                        total_range -= 1

            total_iters += 1
            if total_range == 0 or total_iters >= CONV_ITERATIONS_LIMIT:
                self.ideal_iters_limit = total_iters
                break

    def set_state_details(self):
        self.time_lookup = []
        self.indi_states_lookup = []
        for level in range(2):
            time_list = []
            indi_list = []
            for outer in range(self.size):
                if level == 0:
                    time_list.append([[self.get_cell((i, j)).no_bot_moves for j in range(self.size)] for i in range(self.size)])
                    indi_list.append([[0.0 for j in range(self.size)] for i in range(self.size)])
                else:
                    time_list.append(deepcopy(self.time_lookup))
                    indi_list.append(deepcopy(self.indi_states_lookup))

            self.time_lookup = time_list
            self.indi_states_lookup = indi_list

    def set_calcs_lookup(self):
        self.bot_vs_crew_state = dict()
        self.manhattan_lookup = dict()
        self.bot_moves = dict()
        self.crew_moves = dict()
        iters = 0
        for row in range(self.size):
            for col in range(self.size):
                bot_pos = (row, col)
                if self.get_state(bot_pos) == CLOSED_CELL:
                    continue

                self.bot_vs_crew_state[bot_pos] = list()
                self.manhattan_lookup[bot_pos] = dict()
                bot_movements = self.get_all_moves(bot_pos, ~CLOSED_CELL, False)
                bot_movements.append(bot_pos)
                self.bot_moves[bot_pos] = bot_movements
                crew_states = self.bot_vs_crew_state[bot_pos]
                distances = self.manhattan_lookup[bot_pos]
                for i in range(self.size):
                    for j in range(self.size):
                        crew_pos = (i, j)
                        if self.get_state(crew_pos) == CLOSED_CELL or crew_pos == bot_pos:
                            continue

                        crew_states.append(crew_pos)
                        distances[crew_pos] = get_manhattan_distance(crew_pos, bot_pos)
                        if crew_pos not in self.crew_moves:
                            self.crew_moves[crew_pos] = self.get_all_moves(crew_pos, ~CLOSED_CELL)

                iters += 1

    def get_final_crew_moves(self, bot_pos, crew_pos):
        curr_bot_distance = self.manhattan_lookup[bot_pos][crew_pos]
        new_movements = []
        if curr_bot_distance == 1:
            escape_moves = []
            max_dist = 0
            for crew_move in self.crew_moves[crew_pos]:
                if crew_move == bot_pos:
                    continue

                new_dist =  self.manhattan_lookup[bot_pos][crew_move]
                escape_moves.append((crew_move, new_dist))
                if max_dist < new_dist:
                    max_dist = new_dist

            for escape in escape_moves:
                if escape[1] == max_dist:
                    new_movements.append(escape[0])
        else:
            new_movements = list(self.crew_moves[crew_pos])

        if not len(new_movements):
            new_movements.append(crew_pos)

        return new_movements

    def perform_moves_iteration(self):
        max_iters = self.ideal_iters_limit
        max_iters = 0
        moves_possible = {}
        for iters in range(self.ideal_iters_limit):
            current_iters = 0
            for bot_pos in self.bot_vs_crew_state:
                for crew_pos in self.bot_vs_crew_state[bot_pos]:
                    crew_movements = self.crew_moves[crew_pos]
                    min_max = 0
                    for bot_action in self.bot_moves[bot_pos]:
                        if bot_action == crew_pos:
                            continue

                        new_movements = self.get_final_crew_moves(bot_action, crew_pos)
                        crew_move_prob = 1/len(new_movements)
                        action_time_step = 0.0
                        for crew_move in new_movements:
                            if crew_move == self.teleport_cell:
                                continue

                            action_time_step += self.time_lookup[bot_action[0]][bot_action[1]][crew_move[0]][crew_move[1]]*crew_move_prob

                        if action_time_step > min_max:
                            min_max = action_time_step

                    old_max = self.time_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]]
                    self.time_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]] = 1 + min_max
                    if iters == 0:
                        max_iters += 1
                    elif (min_max + 1) - old_max < 4.3:
                        current_iters += 1

            if current_iters == max_iters:
                break

    def perform_value_iteration(self):
        self.best_policy_lookup = {}
        max_iters = 0
        for iters in range(self.ideal_iters_limit):
            current_iters = 0
            for bot_pos in self.bot_vs_crew_state:
                for crew_pos in self.bot_vs_crew_state[bot_pos]:
                    min_max = self.global_min_max
                    action_val = ()
                    for bot_action in self.bot_moves[bot_pos]:
                        if bot_action == crew_pos:
                            continue

                        new_movements = self.get_final_crew_moves(bot_action, crew_pos)
                        crew_move_prob = 1/len(new_movements)
                        action_value = 0
                        for crew_move in new_movements:
                            if crew_move == self.teleport_cell:
                                continue

                            crew_reward = -1*self.time_lookup[bot_action[0]][bot_action[1]][crew_move[0]][crew_move[1]]
                            new_state_value = self.indi_states_lookup[bot_action[0]][bot_action[1]][crew_move[0]][crew_move[1]]
                            action_value += (crew_move_prob*(crew_reward + new_state_value))

                        if action_value > min_max:
                            min_max = action_value
                            action_val = bot_action


                    old_max = self.indi_states_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]]
                    self.indi_states_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]] = min_max - 1
                    if iters == 0:
                        max_iters += 1
                    elif old_max - (min_max - 1) < CONVERGENCE_LIMIT:
                        if bot_pos not in self.best_policy_lookup:
                            self.best_policy_lookup[bot_pos] = {}

                        bot_policy = self.best_policy_lookup[bot_pos]
                        if crew_pos in bot_policy:
                            bot_policy[crew_pos] = ()

                        bot_policy[crew_pos] = action_val
                        current_iters += 1

            if current_iters == max_iters:
                break


class PARENT_BOT:
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

            if total_iter > 10000:
                self.visualize_grid()
                return total_iter, FAILURE

    def start_data_collection(self,filename):
        total_iter = 0
        if self.ship.get_state(self.local_crew_pos) & TELEPORT_CELL:
            return total_iter,SUCCESS

        while (True):

            self.visualize_grid()
            total_iter += 1
            data = [self.local_bot_pos ,self.local_crew_pos,self.ship.closed_cells]
            print(self.local_bot_pos," ",self.local_crew_pos)
            with open(filename,'a',newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)
            self.move_bot()
            print(self.local_bot_pos)
            if self.move_crew():
                self.visualize_grid()
                data = [self.local_bot_pos ,self.local_crew_pos,self.ship.closed_cells]
                with open(filename,'a',newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(data)
                return total_iter,SUCCESS

            if total_iter > 10000:
                self.visualize_grid()
                return total_iter,FAILURE

class NO_BOT_CONFIG(PARENT_BOT):
    def __init__(self, ship):
        super(NO_BOT_CONFIG, self).__init__(ship)
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

class BOT_CONFIG(PARENT_BOT):
    def __init__(self, ship):
        super(BOT_CONFIG, self).__init__(ship)

    def move_bot(self):
        bot_movements = self.ship.get_all_moves(self.local_bot_pos, OPEN_CELL | TELEPORT_CELL, False)
        bot_movements.append(self.local_bot_pos)
        best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_crew_pos]
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
    if itr % TOTAL_CONFIGS  == 0:
        return NO_BOT_CONFIG(myship)
    else:
        return BOT_CONFIG(myship)

def run_sim(sim_range):
    ship = SHIP()
    ship.perform_initial_calcs()
    avg_moves = [DETAILS() for itr in range(TOTAL_CONFIGS)]
    for _ in sim_range:
        # print(_, end = "\r")
        dest_dist = get_manhattan_distance(ship.crew_pos, ship.teleport_cell)
        for itr in range(TOTAL_CONFIGS):
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

            distance = 0 if test_bot.__class__ is NO_BOT_CONFIG else get_manhattan_distance(ship.bot_pos, ship.crew_pos)
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
    avg_moves = [[DETAILS() for itr in range(TOTAL_CONFIGS)] for _ in range(core_count)]
    with Pool(processes=core_count) as p:
        for layout, final_data in enumerate(p.map(run_sim, arg_data)):
            curr_ship = avg_moves[layout]
            for bot_no, data in enumerate(final_data):
                curr_ship[bot_no].update(data)

        print_header()
        for layout in range(core_count):
            print("Layout no. :: ", layout)
            curr_ship = avg_moves[layout]
            for itr in range(TOTAL_CONFIGS):
                print_data(curr_ship, itr)

def single_sim(total_itr):
    final_data = run_sim(range(0, total_itr))

    print_header(total_itr)
    for itr in range(TOTAL_CONFIGS):
        print_data(final_data, itr, total_itr)

def single_run():
    ship = SHIP()
    ship.perform_initial_calcs()
    ship.print_ship()
    for itr in range(TOTAL_CONFIGS):
        test_bot = bot_fac(itr, ship)
        print(test_bot.start_rescue())
        ship.reset_grid()

def get_data():
    for _ in range(0, 100):  # Adjust the range as needed
        itr = 1
        ship = SHIP()
        ship.perform_initial_calcs()
        test_bot = bot_fac(1, ship)
        end = time()
        print(end - begin)
        # print(test_bot.start_data_collection("output.csv"))
        ship.reset_grid()
        print("data collected for grid no:", _)
        del ship

def create_file():
    filename = "output.csv"
    column_headings = ["Bot_Cell", "Crew_Cell","Closed_Cells"]
    # Check if the file exists and is empty
    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_headings)

if __name__ == '__main__':
    begin = time()
    #single_run()
    # single_sim(1000)
    # run_multi_sim()
    create_file()
    get_data()
    end = time()
    print(end-begin)
