from random import randint, uniform, choice
from math import e as exp
from inspect import currentframe
from time import time
from multiprocessing import Process, Queue, cpu_count

#Constants
CLOSED_CELL = 1
OPEN_CELL = 2
BOT_CELL = 4
CREW_CELL = 8
ALIEN_CELL = 16
GRID_SIZE = 35

TOTAL_ITERATIONS = 10000
MAX_ALPHA_ITERATIONS = 10

X_COORDINATE_SHIFT = [1, 0, 0, -1]
Y_COORDINATE_SHIFT = [0, 1, -1, 0]

ALIEN_ZONE_SIZE = 3 # k - k >= 1, need to determine the large value
ALPHA = 0.02 # avoid alpha > 11 for 35x35
FQ_THRESHOLD = 0.05
INITIAL_BEEP_COUNT = 81

LOG_NONE = 0
LOG_INFO = 1
LOG_DEBUG = 2
LOG_DEBUG_GRID = 3

LOOKUP_E = []
LOOKUP_NOT_E = []

use_version = 1
total_versions = 1 # support till 3 stopped :p


# Common Methods
def get_neighbors(size, cell, grid, filter):
    neighbors = []

    for i in range(4):
        x_cord = cell[0] + X_COORDINATE_SHIFT[i]
        y_cord = cell[1] + Y_COORDINATE_SHIFT[i]
        if (
            (0 <= x_cord < size)
            and (0 <= y_cord < size)
            and (grid[x_cord][y_cord].cell_type & filter)
        ):
            neighbors.append((x_cord, y_cord))

    return neighbors

def print_my_grid(grid):
    print("****************")
    for i, cells in enumerate(grid):
        for j, cell in enumerate(cells):
            print(f"{i}{j}::{cell.cell_type}", end = " ")
        print("")
    print("****************")

class Logger:
    def __init__(self, log_level):
        self.log_level = log_level

    def check_log_level(self, log_level):
        return (self.log_level >= 0) and (log_level <= self.log_level)

    def print(self, log_level, *args):
        if self.check_log_level(log_level):
            print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno, "::", sep="", end="")
            print(*args)

    def print_cell_data(self, log_level, cell, curr_pos):
        if self.check_log_level(log_level):
            print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno, "::", sep="", end="")
            print(f"curr_pos::{curr_pos}, cell_cord::{cell.cord}, cell_distance::{cell.bot_distance}, cell.probs.crew_prob::{cell.probs.crew_prob}, cell.probs.beep_given_crew::{cell.probs.beep_given_crew}, cell.probs.no_beep_given_crew::{cell.probs.no_beep_given_crew}, cell.probs.crew_and_beep::{cell.probs.crew_and_beep}, cell.probs.crew_and_no_beep::{cell.probs.crew_and_no_beep}")

    def print_grid(self, log_level, grid):
        if not self.check_log_level(log_level):
            return

        print("****************")
        for i, cells in enumerate(grid):
            for j, cell in enumerate(cells):
                print(f"{i}{j}::{cell.cell_type}", end = " ")
            print("")
        print("****************")

class Cell_Probs:
    def __init__(self):
        self.crew_prob = 0
        self.crew_given_beep = 0
        self.crew_given_no_beep = 0
        self.crew_and_beep = 0
        self.crew_and_no_beep = 0


class Crew_Probs:
    def __init__(self, ship):
        self.beep_prob = self.no_beep_prob = 0
        self.crew_cells = list(ship.open_cells)
        self.crew_cells.append(ship.crew)
        self.is_beep_recv = False
        self.beep_count = 0
        self.max_beeps = INITIAL_BEEP_COUNT


class Cell:
    def __init__(self, row, col, cell_type = OPEN_CELL):
        self.cell_type = cell_type
        self.bot_distance = 0
        self.probs = Cell_Probs()
        self.cord = (row, col)
        self.crew_distance = 0
        self.prob_hearing_beep = 0
        self.within_detection_zone = False


class Ship:
    def __init__(self, size, log_level = LOG_INFO):
        self.size = size
        self.grid = [[Cell(i, j, CLOSED_CELL) for j in range(size)] for i in range(size)]
        self.open_cells = []
        self.logger = Logger(log_level)
        self.isBeep = 0
        self.bot = (0, 0)
        self.crew = (0, 0)
        self.alien = (0, 0)

        #closed grid formation...
        self.generate_grid()

    def count_open_neighbors(self, row, col):
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        count = 0
        for i in range(4):
            x_shift = row + X_COORDINATE_SHIFT[0]
            y_shift = col + Y_COORDINATE_SHIFT[0]
            if 0 <= x_shift < self.size and 0 <= y_shift < self.size and (self.grid[x_shift][y_shift].cell_type & CLOSED_CELL):
                count += 1
        return count
    
    def generate_grid(self):
        self.assign_start_cell()
        self.unblock_closed_cells()
        self.unblock_dead_ends()

    def assign_start_cell(self):
        random_row = randint(0, self.size - 1)
        random_col = randint(0, self.size - 1)
        self.grid[random_row][random_col].cell_type = OPEN_CELL

    def unblock_closed_cells(self):
        available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)
        while available_cells:
            closed_cell = choice(available_cells)
            self.grid[closed_cell[0]][closed_cell[1]].cell_type = OPEN_CELL
            available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)

    def unblock_dead_ends(self):
        dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
        half_len = len(dead_end_cells)/2

        while half_len > 0:

            dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
            half_len -= 1
            # if len(dead_end_cells) == 0: continue
            dead_end_cell = choice(dead_end_cells)
            closed_neighbors = get_neighbors(
                self.size, dead_end_cell, self.grid, CLOSED_CELL
            )

            random_cell = choice(closed_neighbors)
            self.grid[random_cell[0]][random_cell[1]].cell_type = OPEN_CELL

    def cells_with_one_open_neighbor(self, cell_type):
        results = []
        for row in range(self.size):
            for col in range(self.size):
                if ((self.grid[row][col].cell_type & cell_type) and
                    len(get_neighbors(
                        self.size, (row, col), self.grid, OPEN_CELL
                    )) == 1):
                    results.append((row, col))
        return results

    def place_players(self):
        # self.bot = (0, 0)
        while(True):
            self.bot = (randint(0, self.size -1), randint(0, self.size -1))
            if (self.grid[self.bot[0]][self.bot[1]]).cell_type & OPEN_CELL:
                break

        while(True):
            # self.crew = (self.size - 1, self.size - 1)
            self.crew = (randint(0, self.size -1), randint(0, self.size -1))
            if self.bot != self.crew and (self.grid[self.crew[0]][self.crew[1]]).cell_type & OPEN_CELL:
                break

        self.set_cell_details()
        self.grid[self.bot[0]][self.bot[1]].cell_type = BOT_CELL
        self.grid[self.crew[0]][self.crew[1]].cell_type = CREW_CELL

        self.place_aliens()

    def set_cell_details(self, reset = False):
        for i, cells in enumerate(self.grid):
            for j, cell in enumerate(cells):
                if cell.cord != self.bot and cell.cord != self.crew and cell.cell_type != CLOSED_CELL:
                    cell.cell_type = OPEN_CELL
                    self.open_cells.append((i, j))

                if cell.cell_type & (OPEN_CELL | BOT_CELL | CREW_CELL):
                    cell.crew_distance = abs(self.crew[0] - i) + abs(self.crew[1] - j)
                    cell.prob_hearing_beep = LOOKUP_E[cell.crew_distance]
                    cell.cord = (i, j)

    def crew_beep(self, cell):
        self.isBeep = uniform(0, 1)
        if self.isBeep <= self.grid[cell[0]][cell[1]].prob_hearing_beep:
            return True
        return False
    
    def place_aliens(self):
        # print('Bot cell: ', self.bot)
        cells_within_zone = self.get_detection_zone(self.bot)

        while(True):
            self.alien = (randint(0, self.size -1), randint(0, self.size -1))
            if ((self.alien not in cells_within_zone) and
                ((self.grid[self.alien[0]][self.alien[1]]).cell_type & OPEN_CELL)):
                break

        # print('Alien cell: ', self.alien)
        self.grid[self.alien[0]][self.alien[1]].cell_type = ALIEN_CELL

    
    def get_detection_zone(self, cell):
        k = ALIEN_ZONE_SIZE

        cells_within_zone = []
        min_row = max(0, cell[0] - k)
        max_row = min(self.size - 1, cell[0] + k)
        min_col = max(0, cell[1] - k)
        max_col = min(self.size - 1, cell[1] + k)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell = self.grid[row][col]
                cell.within_detection_zone = True
                cells_within_zone.append((row, col))
        
        # print('Cells within zone: ', cells_within_zone)

        return cells_within_zone
    
    def reset_detection_zone(self, curr_cell):
        cells_within_zone = self.get_detection_zone(curr_cell)
        
        # Reset cells outside detection zone to false
        for i, cells in enumerate(self.grid):
            for j, cell in enumerate(cells):
                if cell not in cells_within_zone: cell.within_detection_zone = False

        return cells_within_zone

    def reset_grid(self):
        self.open_cells = []
        self.set_cell_details(True)
        self.grid[self.bot[0]][self.bot[1]].cell_type = BOT_CELL
        self.grid[self.crew[0]][self.crew[1]].cell_type = CREW_CELL


class SearchAlgo:
    def __init__(self, ship, log_level):
        self.ship = ship
        self.curr_pos = ship.bot
        self.logger = Logger(log_level)

    def search_path(self, dest_cell):
        bfs_queue = []
        visited_cells = set()
        bfs_queue.append((self.curr_pos, [self.curr_pos]))

        while bfs_queue:
            current_cell, path_traversed = bfs_queue.pop(0)
            if current_cell == dest_cell:
                return path_traversed
            elif (current_cell in visited_cells):
                continue

            visited_cells.add(current_cell)
            neighbors = get_neighbors(self.ship.size, current_cell, self.ship.grid, (OPEN_CELL | CREW_CELL))
            for neighbor in neighbors:
                if (neighbor not in visited_cells):
                    bfs_queue.append((neighbor, path_traversed + [neighbor]))

        return [] #God forbid, this should never happen


class ParentBot(SearchAlgo):
    def __init__(self, ship, log_level):
        super(ParentBot, self).__init__(ship, log_level)
        self.crew_probs = Crew_Probs(ship)
        self.traverse_path = []
        self.pred_crew_cells = []
        self.is_keep_moving = self.is_inital_calc_done = False
        self.recalc_pred_cells = True
        self.path_pos = 0
        self.logger.print_grid(LOG_DEBUG_GRID, self.ship.grid)

    def update_cell_mov_vals(self, cell, cell_cord):
        cell.bot_distance = abs(self.curr_pos[0] - cell_cord[0]) + abs(self.curr_pos[1] - cell_cord[1])
        cell.probs.beep_given_crew = LOOKUP_E[cell.bot_distance]
        cell.probs.no_beep_given_crew = LOOKUP_NOT_E[cell.bot_distance]
        cell.probs.crew_and_beep = cell.probs.beep_given_crew * cell.probs.crew_prob
        cell.probs.crew_and_no_beep = cell.probs.no_beep_given_crew * cell.probs.crew_prob
        self.crew_probs.beep_prob += cell.probs.crew_and_beep
        self.crew_probs.no_beep_prob += cell.probs.crew_and_no_beep
        return cell

    def calc_initial_probs(self):
        if (self.is_inital_calc_done):
            return

        crew_cell_size = len(self.crew_probs.crew_cells)
        for cell_cord in self.crew_probs.crew_cells:
            cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
            cell.probs.crew_prob = 1/crew_cell_size
            cell = self.update_cell_mov_vals(cell, cell_cord)

        self.is_inital_calc_done = True

    """
        Ideally it is better to move the bot in the direction of the highest prob
        To do this, pred_crew_cells should be sorted based on probabilty
        Remember, we are not taking into account where the alien will be here!!
    """
    def move_bot(self):
        prob_crew_cell = ()
        if len(self.traverse_path) == 0 and len(self.pred_crew_cells) != 0:
            prob_crew_cell = self.pred_crew_cells.pop(0)
            self.traverse_path = self.search_path(prob_crew_cell)
            self.traverse_path.pop(0)
            self.logger.print(LOG_DEBUG, f"New path found, {self.traverse_path}. Pending cells to explore, {self.pred_crew_cells}")

        if len(self.traverse_path) == 0: # somehow there is one edge case?
            self.logger.print(LOG_DEBUG, f"Bot in {self.curr_pos} with crew cells {self.crew_probs.crew_cells} and last prob_crew_cell was {prob_crew_cell}")
            self.logger.print(LOG_DEBUG, f"Bot started {self.ship.bot} with crew at {self.ship.crew}")
            self.logger.print(LOG_DEBUG, f"pred_crew_cells::{self.pred_crew_cells}")
            return False

        self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].cell_type = OPEN_CELL
        old_pos = self.curr_pos
        self.curr_pos = self.traverse_path.pop(0)
        if (self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].cell_type & CREW_CELL):
            self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].cell_type |= BOT_CELL
        else:
            self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].cell_type = BOT_CELL

        self.is_keep_moving = True if len(self.traverse_path) or len(self.pred_crew_cells) else False
        self.recalc_pred_cells = not self.is_keep_moving
        self.logger.print(LOG_DEBUG, f"Bot{old_pos} has moved to {self.curr_pos}")
        self.logger.print_grid(LOG_DEBUG_GRID, self.ship.grid)
        return True

class Bot_1(ParentBot):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_1, self).__init__(ship, log_level)

    def calc_crew_probs(self, observed_fq, bot_moved, is_move_bot):
        self.calc_initial_probs()

        beep_prob = no_beep_prob = 0
        observed_fq = round(observed_fq, 2)
        threshold = FQ_THRESHOLD * (self.crew_probs.max_beeps / INITIAL_BEEP_COUNT)
        min_prob = 0 if (observed_fq - threshold < 0) else (observed_fq - threshold)
        max_prob = 1 if (observed_fq + threshold > 1) else (observed_fq + threshold)
        pred_crew_cells = []
        all_probs = set()
        pred_crew_cells_2 = [] #take first few cells (what is few??)
        is_recalc_cell_prob = False

        # reason for having 2 prob cells is because the cell with the highest prob can be the adjacent cell and this can be ignored if no beep was recieved
        if (len(self.crew_probs.crew_cells) == 0):
             self.logger.print(LOG_NONE, f"Bot in {self.curr_pos} has no crew cells!!!")
             self.logger.print(LOG_NONE, f"Bot started {self.ship.bot} with crew at {self.ship.crew}")
             self.logger.print(LOG_NONE, f"pred_crew_cells::{self.pred_crew_cells}")
             exit()

        if not self.crew_probs.is_beep_recv:
            neighbors = get_neighbors(self.ship.size, self.curr_pos, self.ship.grid, (OPEN_CELL | CREW_CELL))
            neighbors_in_crew = [neighbor for neighbor in neighbors if neighbor in self.crew_probs.crew_cells]
            if (len(neighbors_in_crew)):
                for neighbor in neighbors_in_crew:
                    if neighbor in self.pred_crew_cells:
                        self.pred_crew_cells.remove(neighbor)
                    self.crew_probs.crew_cells.remove(neighbor)
                    if (is_move_bot):
                        self.is_keep_moving = True if len(self.traverse_path) or len(self.pred_crew_cells) else False
                        self.recalc_pred_cells = not self.is_keep_moving
                is_recalc_cell_prob = True # recalc for all cells, calc is similar for each cell in a way...
                self.logger.print(LOG_DEBUG, f"Following cells {neighbors_in_crew}, were removed from crew cells {self.crew_probs.crew_cells} and pred cells {self.pred_crew_cells}")

        if (bot_moved) or is_recalc_cell_prob:
            self.crew_probs.beep_prob = 0
            self.crew_probs.no_beep_prob = 0
            for cell_cord in self.crew_probs.crew_cells:
                cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
                cell = self.update_cell_mov_vals(cell, cell_cord)

        self.logger.print(LOG_DEBUG, f"is_beep_recv::{self.crew_probs.is_beep_recv}, observed_fq::{observed_fq}, min_prob::{min_prob}, max_prob::{max_prob}")
        self.logger.print(LOG_DEBUG, f"beep_prob::{self.crew_probs.beep_prob}, no_beep_prob::{self.crew_probs.no_beep_prob}")

        for cell_cord in self.crew_probs.crew_cells:
            is_removed = True
            cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
            self.logger.print_cell_data(LOG_DEBUG, cell, self.curr_pos)
            if not (self.crew_probs.beep_prob):
                self.logger.print_cell_data(LOG_NONE, cell, self.curr_pos)
                self.logger.print(LOG_NONE, f"is_beep_recv::{self.crew_probs.is_beep_recv}, observed_fq::{observed_fq}, min_prob::{min_prob}, max_prob::{max_prob}")
                self.logger.print(LOG_NONE, f"beep_prob::{self.crew_probs.beep_prob}, no_beep_prob::{self.crew_probs.no_beep_prob}")
                self.logger.print(LOG_NONE, f"Bot in {self.curr_pos} has updated crew cells to be, {self.crew_probs.crew_cells}. The pred cells is {self.pred_crew_cells}, with traverse path {self.traverse_path}")

            cell.probs.crew_given_beep = (cell.probs.crew_and_beep) / self.crew_probs.beep_prob
            if (self.crew_probs.no_beep_prob != 0):
                cell.probs.crew_given_no_beep = (cell.probs.crew_and_no_beep) / self.crew_probs.no_beep_prob
            cell.probs.crew_prob = cell.probs.crew_given_beep if self.crew_probs.is_beep_recv else cell.probs.crew_given_no_beep
            is_removed = False
            # not good cause better to towards the cell with the prob of a crew in it???
            if (cell.probs.beep_given_crew >= min_prob and cell.probs.beep_given_crew <= max_prob):
                # pred_crew_cells.append(cell_cord)
                pred_crew_cells.append((cell_cord, cell.probs.crew_given_beep)) # move to all cells we have deemed worthy :p

            if cell.probs.crew_given_beep not in all_probs:
                all_probs.add(cell.probs.crew_given_beep)

            # if not (cell.probs.crew_prob):
            #     self.logger.print_cell_data(LOG_NONE, cell, self.curr_pos)
            #     self.logger.print(LOG_NONE, f"is_beep_recv::{self.crew_probs.is_beep_recv}, observed_fq::{observed_fq}, min_prob::{min_prob}, max_prob::{max_prob}")
            #     self.logger.print(LOG_NONE, f"beep_prob::{self.crew_probs.beep_prob}, no_beep_prob::{self.crew_probs.no_beep_prob}")
                # self.logger.print(LOG_NONE, f"Bot in {self.curr_pos} has updated crew cells to be, {self.crew_probs.crew_cells}. The pred cells is {self.pred_crew_cells}, with traverse path {self.traverse_path}")
            cell.probs.crew_and_beep = cell.probs.beep_given_crew * cell.probs.crew_prob
            cell.probs.crew_and_no_beep = cell.probs.no_beep_given_crew * cell.probs.crew_prob
            beep_prob += cell.probs.crew_and_beep
            no_beep_prob += cell.probs.crew_and_no_beep
            self.logger.print_cell_data(LOG_DEBUG, cell, self.curr_pos)

        if self.recalc_pred_cells and not(is_move_bot):
            # self.pred_crew_cells = list(pred_crew_cells)
            # pred_crew_cells = sorted(pred_crew_cells, key=lambda crew_prob : crew_prob[1], reverse = True) # could make this slower "potentitally??"
            all_probs = sorted(all_probs, reverse=True)
            self.pred_crew_cells = list()
            for cell in pred_crew_cells:
                max_range = 5 if len(all_probs) > 5 else len(all_probs)
                for i in range(max_range):
                    if cell[1] == all_probs[i]:
                        self.pred_crew_cells.append(cell[0])

        self.crew_probs.beep_prob = beep_prob
        self.crew_probs.no_beep_prob = no_beep_prob
        self.logger.print(LOG_DEBUG, f"beep_prob::{self.crew_probs.beep_prob}, no_beep_prob::{self.crew_probs.no_beep_prob}")
        self.logger.print(LOG_DEBUG, f"Bot in {self.curr_pos} has updated crew cells to be, {self.crew_probs.crew_cells}. The pred cells is {self.pred_crew_cells}, with traverse path {self.traverse_path}")

    def is_rescued(self, init_distance, total_iter, idle_steps):
        if (self.curr_pos == self.ship.crew):
            self.logger.print(LOG_INFO, f"Congrats, you found a crew member who was initially {init_distance} steps away from you after {total_iter} steps. You moved {total_iter - idle_steps} steps, and waited for {idle_steps} steps")
            return True
        elif (self.curr_pos in self.crew_probs.crew_cells):
            self.crew_probs.crew_cells.remove(self.curr_pos)
            self.logger.print(LOG_DEBUG, f"Removing current position{self.curr_pos} from list of probable crew cells{self.crew_probs.crew_cells}")
            self.crew_probs.beep_prob -= self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].probs.crew_and_beep
            self.crew_probs.no_beep_prob -= self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].probs.crew_and_no_beep

        return False

    def start_rescue(self):
        bot_moved = is_move_bot = False
        idle_steps = total_iter = 0
        beep_counter = 1
        init_distance = self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].crew_distance
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew}")
        if (self.is_rescued(init_distance, total_iter, idle_steps)):
            return (init_distance, total_iter, idle_steps)

        while (True): # Keep trying till you find the crew
            total_iter += 1

            self.crew_probs.is_beep_recv = self.ship.crew_beep(self.curr_pos)
            if self.crew_probs.is_beep_recv:
                self.crew_probs.beep_count += 1
            self.logger.print(LOG_DEBUG, self.crew_probs.beep_count, beep_counter)

            self.calc_crew_probs((self.crew_probs.beep_count/beep_counter), bot_moved, self.is_keep_moving or is_move_bot)
            if (self.is_keep_moving or is_move_bot) and self.move_bot():
                if (self.is_rescued(init_distance, total_iter, idle_steps)):
                    return (init_distance, total_iter, idle_steps)

                self.crew_probs.beep_count = 0
                self.crew_probs.max_beeps = INITIAL_BEEP_COUNT
                beep_counter = 1
                is_move_bot = False
                bot_moved = True
            else:
                is_move_bot = False
                bot_moved = False
                idle_steps += 1

            beep_counter += 1
            if beep_counter >= (self.crew_probs.max_beeps):
                if len(self.pred_crew_cells):
                    is_move_bot = True
                else:
                    self.crew_probs.max_beeps *= 2


def update_lookup(alpha):
    global LOOKUP_E, LOOKUP_NOT_E, ALPHA
    ALPHA = alpha
    LOOKUP_E = [pow(exp, (-1*ALPHA*(i - 1))) for i in range(GRID_SIZE*2 + 1)]
    LOOKUP_NOT_E = [(1-LOOKUP_E[i]) for i in range(GRID_SIZE*2 + 1)]

def run_test(log_level = LOG_INFO):
    update_lookup(ALPHA)
    ship = Ship(GRID_SIZE)
    ship.place_players()
    # print_my_grid(ship.grid)
    use_version = 1
    # bot_1 = Bot_1(ship, log_level)
    # bot_1.start_rescue()
    # del bot_1
    del ship

def run_sim(my_range, queue, alpha):
    update_lookup(alpha)
    temp_data_set = [[0.0 for i in range(4)] for j in range(total_versions)]
    space_itr = round((my_range[0]/100) + 1)
    for itr in my_range:
        print(itr+1, end='\r')
        ship = Ship(GRID_SIZE)
        ship.place_players()
        for i in range(total_versions):
            use_version = i + 1
            bot_1 = Bot_1(ship)
            begin = time()
            ret_vals = bot_1.start_rescue()
            end = time()
            for j in range(3):
                temp_data_set[use_version - 1][j] += ret_vals[j]
            temp_data_set[use_version - 1][3] += (end-begin)
            ship.reset_grid()
            del bot_1
        del ship

    queue.put(temp_data_set)

def run_multi_sim(alpha, is_print = False):
    begin = time()
    data_set = [[0.0 for i in range(4)] for j in range(total_versions)]
    processes = []
    queue = Queue()
    if (is_print):
        print(f"Iterations begin...")
    core_count = cpu_count()
    total_iters = round(TOTAL_ITERATIONS/core_count)
    actual_iters = total_iters * core_count
    for itr in range(core_count):
        p = Process(target=run_sim, args=(range(itr*total_iters, (itr+1)*total_iters), queue, alpha))
        processes.append(p)
        p.start()

    for proc in processes:
        proc.join()
        temp_data_set = queue.get()
        for i in range(total_versions):
            for j in range(4):
                data_set[i][j] += temp_data_set[i][j]

    for i in range(total_versions):
        data_set[i][0] = data_set[i][0]/actual_iters
        data_set[i][1] = data_set[i][1]/actual_iters
        data_set[i][2] = data_set[i][2]/actual_iters
        data_set[i][3] = data_set[i][3]/actual_iters
    end = time()
    if (is_print):
        print()
        print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {alpha}")
        print(f"distance\ttotal iter\tidle steps\tsteps moved\ttime taken")
        for i in range(0,1):
            print(f"{data_set[i][0]}\t{data_set[i][1]}\t{data_set[i][2]}\t{data_set[i][1]-data_set[i][2]}\t{data_set[i][3]}")
    else:
        print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {alpha}")

    del queue
    del processes
    return data_set[0]

def compare_multiple_alpha():
    global ALPHA
    begin = time()
    alpha_map = {}
    for itr in range(MAX_ALPHA_ITERATIONS):
        alpha_map[ALPHA] = run_multi_sim(ALPHA)
        ALPHA = round(ALPHA + 0.2, 2)
    end = time()
    print(f"It took {end - begin} seconds to complete computation")
    for key, value in alpha_map.items():
        print(key, value)

if __name__ == '__main__':
    run_test()
    # run_multi_sim(ALPHA, True)
    # compare_multiple_alpha()
