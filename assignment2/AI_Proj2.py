from random import randint, uniform, choice
from math import e as exp, ceil, sqrt
from inspect import currentframe
from matplotlib import pyplot
from time import time
from multiprocessing import Process, Queue, cpu_count
from itertools import permutations

#Constants
CLOSED_CELL = 1
OPEN_CELL = 2
BOT_CELL = 4
CREW_CELL = 8
ALIEN_CELL = 16
BOT_CAUGHT_CELL = 32
PRED_ALIEN_CELL = 64
BOT_MOVEMENT_CELLS = OPEN_CELL | CREW_CELL | ALIEN_CELL
ALIEN_MOVEMENT_CELLS = CREW_CELL | OPEN_CELL | BOT_CELL
GRID_SIZE = 35
ADDITIVE_VALUE = 1e-6

BOT_SUCCESS = 1
BOT_FAILED = 2
BOT_STUCK = 3

X_COORDINATE_SHIFT = [1, 0, 0, -1]
Y_COORDINATE_SHIFT = [0, 1, -1, 0]

ALIEN_ZONE_SIZE = 1 # k - k >= 1, need to determine the large value
ALPHA = 0.05 # avoid large alpha at the cost of performance
IDLE_BEEP_COUNT = 10
TOTAL_UNSAFE_CELLS = 5

TOTAL_ITERATIONS = 1000
MAX_ALPHA_ITERATIONS = 10
ALPHA_STEP_INCREASE = 0.05
TOTAL_BOTS = 2

LOG_NONE = 0
LOG_DEBUG_ALIEN = 0.5
LOG_INFO = 1
LOG_DEBUG = 2
LOG_DEBUG_GRID = 3
IGNORE_GRID_DEBUG = True

LOOKUP_E = []
LOOKUP_NOT_E = []

ALIEN_NOT_PRESENT = 0
ALIEN_PRESENT = 1

ONE_ALIEN = 1
TWO_ALIENS = 2


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

def get_manhattan_distance(cell_1, cell_2):
    return abs(cell_1[0] - cell_2[0]) + abs(cell_1[1] - cell_2[1])

def check_for_failure(bot, condition):
    if condition == 1 and (len(bot.crew_search_data.crew_cells) == 0):
        print()
        print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno)
        bot.logger.print(LOG_NONE, f"curr pos {bot.curr_pos}, crew_1 {bot.ship.crew_1}, crew_2 {bot.ship.crew_2}")
        bot.logger.print(LOG_NONE, f"No crew cells!!!")
        bot.logger.print(LOG_NONE, f"pred_crew_cells::{bot.pred_crew_cells}")
        bot.logger.print(LOG_NONE, f"path_traversed::{bot.path_traversed}")
        exit()
    elif condition == 2 and (not bot.crew_search_data.beep_prob or not bot.crew_search_data.normalize_probs):
        print()
        print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno)
        for cell_cord in bot.crew_search_data.crew_cells:
            cell = bot.ship.get_cell(cell_cord)
            bot.logger.print_cell_data(LOG_NONE, cell.crew_probs, cell.cord, bot.curr_pos)

        bot.logger.print(LOG_NONE, f"Followup list {bot.crew_search_data.followup_list}")
        bot.logger.print(LOG_NONE, f"is_beep_recv::{bot.crew_search_data.is_beep_recv}, {bot.ship.isBeep}, normalize_probs::{bot.crew_search_data.normalize_probs}, beep_prob::{bot.crew_search_data.beep_prob}")
        bot.logger.print(LOG_NONE, f"Bot in {bot.curr_pos} has to find crew({bot.total_crew_to_save}) {bot.ship.crew_1, bot.ship.crew_2} with pending crew {bot.pending_crew}")
        # bot.logger.print(LOG_NONE, f"path_traversed {bot.path_traversed}")
        bot.logger.print(LOG_NONE, f"The crew cells are {bot.crew_search_data.crew_cells}. The pred cells is {bot.pred_crew_cells}, with traverse path {bot.traverse_path}")
        exit()
    elif condition == 3 and not bot.crew_search_data.beep_prob:
        print()
        bot.logger.print(LOG_NONE, f"is_beep_recv::{bot.crew_search_data.is_beep_recv}, {bot.ship.isBeep}")
        bot.logger.print(LOG_NONE, f"Bot in {bot.curr_pos} has to find crew({bot.total_crew_to_save}) {bot.ship.crew_1, bot.ship.crew_2} with pending crew {bot.pending_crew}")
        bot.logger.print(LOG_NONE, f"path_traversed {bot.path_traversed}")
        bot.logger.print(LOG_NONE, f"The crew cells are {bot.crew_search_data.crew_cells}. The pred cells is {bot.pred_crew_cells}, with traverse path {bot.traverse_path}")
        bot.crew_search_data.print_map(bot.curr_pos)

# used for debugging (mostly...)
class Logger:
    def __init__(self, log_level):
        self.log_level = log_level

    def check_log_level(self, log_level):
        return (self.log_level >= 0) and (log_level <= self.log_level)

    def print(self, log_level, *args):
        if self.check_log_level(log_level):
            print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno, "::", sep="", end="")
            print(*args)

    def print_cell_data(self, log_level, crew_probs, cords, curr_pos):
        if self.check_log_level(log_level):
            print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno, "::", sep="", end="")
            print(f"curr_pos::{curr_pos}, cell_cord::{(cords.cell_1, cords.cell_2) if type(cords) is Cell_Pair else cords}, cell_distance::{crew_probs.bot_distance}, crew_prob::{crew_probs.crew_prob}, beep_given_crew::{crew_probs.beep_given_crew}, no_beep_given_crew::{crew_probs.no_beep_given_crew}, crew_and_beep::{crew_probs.crew_and_beep}, crew_and_no_beep::{crew_probs.crew_and_no_beep}")

    def print_grid(self, grid, log_level = LOG_DEBUG_GRID):
        if not self.check_log_level(log_level):
            return

        print("****************")
        for i, cells in enumerate(grid):
            for j, cell in enumerate(cells):
                print("%10s" % (str(i) + str(j) + "::" + str(cell.cell_type)), end = " ")
            print("")
        print("****************")

    def print_crew_data(self, curr_pos, ship, crew_search_data, beep_count, is_beep_recv):
        print(currentframe().f_back.f_back.f_code.co_name, "::", currentframe().f_back.f_back.f_lineno, curr_pos, beep_count, is_beep_recv)
        print("%8s %27s %3s %27s %27s %27s %27s %27s %27s %27s %27s %20s" % ("cell", "p(C)", "B_D", "p(B|C)", "p(nB|c)", "p(C,B)", "p(C,nB)", "p(C|B)", "p(C|nB)", "p(B)",  "p(nB)",  "norm p(C)"))
        for cell_cord in crew_search_data.crew_cells:
            crew_probs = ship.get_cell(cell_cord).crew_probs
            print("%8s %27s %3s %27s %27s %27s %27s %27s %27s %27s %27s %20s" % (cell_cord, crew_probs.crew_prob, crew_probs.bot_distance, crew_probs.beep_given_crew, crew_probs.no_beep_given_crew, crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_given_beep, crew_probs.crew_given_no_beep, crew_search_data.beep_prob,  crew_search_data.no_beep_prob,  crew_search_data.normalize_probs))

    def print_all_crew_data(self, log_level, bot):
        if not self.check_log_level(log_level):
            return

        self.print_crew_data(bot.curr_pos, bot.ship, bot.crew_search_data, bot.total_beep, bot.is_beep_recv)

    def print_crew_probs(self, grid, is_beep_recv, curr_pos):
        if IGNORE_GRID_DEBUG and not self.check_log_level(LOG_DEBUG_GRID):
            return

        prob_grid = []
        prob_spread = list()
        for cells in grid:
            prob_cell = []
            for cell in cells:
                if cell.cell_type == CLOSED_CELL:
                    prob_cell.append(float('nan'))
                elif cell.cell_type == (BOT_CELL|CREW_CELL):
                    prob_cell.append(1)
                else:
                    prob_cell.append(cell.crew_probs.crew_prob)
                    if not cell.crew_probs.crew_prob in prob_spread:
                        prob_spread.append(cell.crew_probs.crew_prob)
            prob_grid.append(prob_cell)

        prob_spread.sort()
        max_len = len(prob_spread) - 1
        prob_grid[curr_pos[0]][curr_pos[1]] = prob_spread[max_len]

        pyplot.figure(figsize=(10,10))
        pyplot.colorbar(pyplot.imshow(prob_grid, vmin=prob_spread[0], vmax=prob_spread[max_len]))
        pyplot.title("Beep recv" if is_beep_recv else "Beep not recv")
        pyplot.show()

# Modularizing our knowledge base for readability
class One_Alien_Evasion_Data:
    def __init__(self, ship):
        self.ship = ship
        self.init_alien_cells = list(self.ship.initial_alien_cells)
        self.alien_cells = list(self.ship.open_cells)
        self.alien_cells.extend([self.ship.crew_1, self.ship.crew_2, self.ship.bot])
        self.present_alien_cells = []
        self.is_beep_recv = False
        self.beep_prob = 0
        self.beep_count = 0

        self.init_alien_cell_size = len(self.init_alien_cells)
        self.alien_cell_size = len(self.alien_cells)
        self.init_alien_calcs()

    def init_alien_calcs(self):
        for cell_cord in self.init_alien_cells:
            cell = self.ship.get_cell(cell_cord)
            cell.alien_probs.alien_prob = 1/self.init_alien_cell_size
    
    def reset_alien_calcs(self):
        for cell_cord in self.alien_cells:
            cell = self.ship.get_cell(cell_cord)
            cell.alien_probs.alien_prob = 0 if cell.within_detection_zone else 1/self.init_alien_cell_size


# class Two_Alien_Evasion_Data(One_Alien_Evasion_Data):
#     def __init__(self, ship):
#         self.alien_cells_pair = dict()
#         super(Two_Alien_Evasion_Data, self).__init__(ship)

#     def init_alien_calcs(self):
#         self.alien_pair_len = (self.init_alien_cell_size * (self.init_alien_cell_size - 1)) # Considering all 800 x 800 cells
#         for i, key in enumerate(self.alien_cells):
#             if i == self.alien_cell_size - 1:
#                 break

#             self.alien_cells_pair[key] = list()
#             print(key, "::", end = " ")
#             for j in range(self.alien_cell_size):
#                 if i == j: continue
#                 cell_pair = Alien_Cell_Pair(key, self.alien_cells[j], self.ship)
#                 cell_1, cell_2 = cell_pair.cell_1, cell_pair.cell_2
#                 cell_pair.alien_probs.alien_prob = 1/self.alien_pair_len if (cell_1 in self.init_alien_cells and cell_2 in self.init_alien_cells) else ALIEN_NOT_PRESENT
#                 self.alien_cells_pair[(cell_1.cord, cell_2.cord)] = cell_pair

    # def add_cell_pair(self, cell_1, cell_2):
    # def remove_cell_pair(self, cell_pair):
    # def list_all_pairs(self):


class Alien_Probs:
    def __init__(self):
        self.alien_prob = ALIEN_NOT_PRESENT
        self.alien_and_beep = ALIEN_NOT_PRESENT

class Crew_Probs: # contains all the prob of crew in each cell
    def __init__(self):
        self.bot_distance = 0 # distance of curr cell to bot
        self.crew_prob = 0.0 # p(c)
        self.beep_given_crew = 0.0 # p(b|c)
        self.no_beep_given_crew = 0.0 # p(¬b|c)
        self.crew_given_beep = 0.0 # p(c|b)
        self.crew_given_no_beep = 0.0 # p(c|¬b)
        self.crew_and_beep = 0.0 # p (c,b)
        self.crew_and_no_beep = 0.0 # p(c,¬b)
        self.track_beep = list((0, 0))

    def update_crew_probs(self, crew_search_data):
        self.crew_given_beep = (self.crew_and_beep) / crew_search_data.beep_prob
        if (crew_search_data.no_beep_prob != 0):
            self.crew_given_no_beep = (self.crew_and_no_beep) / crew_search_data.no_beep_prob

        self.crew_prob = self.crew_given_beep if crew_search_data.is_beep_recv else self.crew_given_no_beep
        self.crew_and_beep = self.beep_given_crew * self.crew_prob
        self.crew_and_no_beep = self.no_beep_given_crew * self.crew_prob
        crew_search_data.normalize_probs += self.crew_prob

class Beep:
    def __init__(self):
        self.crew_1_dist = self.crew_2_dist = 0 # distance of current cell to crew
        self.c1_beep_prob = self.c2_beep_prob = 0 # probability of hearing the crews beep from this cell

class Cell: # contains the detail inside each cell (i, j)
    def __init__(self, row, col, cell_type = OPEN_CELL):
        self.cell_type = cell_type # static constant
        self.within_detection_zone = False
        self.crew_probs = Crew_Probs()
        self.alien_probs = Alien_Probs()
        self.listen_beep = Beep()
        self.cord = (row, col) # coordinate of this cell
        self.adj_cells = []

class Cell_Pair:
    def __init__(self, cell_1, cell_2, ship):
        self.cell_1 = ship.get_cell(cell_1)
        self.cell_2 = ship.get_cell(cell_2)
        self.cells = [self.cell_1, self.cell_2]
        self.init_probs()

    def init_probs(self):
        self.crew_probs = Crew_Probs()
        self.cell_distances = 0

class Alien_Cell_Pair(Cell_Pair):
    def __init__(self, cell_1, cell_2, ship):
        super(Alien_Cell_Pair, self).__init__(cell_1, cell_2, ship)

    def init_probs(self):
        self.alien_probs = Alien_Probs()

class Crew_Search_Data:
    def __init__(self):
        self.beep_prob = 0.0  # p(b) -> normalized for hearing beeps
        self.no_beep_prob = 0.0 # p(¬b) -> normalized for hearing no beeps
        self.normalize_probs = 0.0 # probs will be reduced from this to normalize them
        self.beep_count = 0
        self.is_beep_recv = False

    def set_all_probs(self, beep_prob = 0.0, no_beep_prob = 0.0, norm_prob = 0.0):
        self.beep_prob = beep_prob
        self.no_beep_prob = no_beep_prob
        self.normalize_probs = norm_prob

    def update_all_probs(self, beep_prob = 0.0, no_beep_prob = 0.0, norm_prob = 0.0):
        self.beep_prob += beep_prob
        self.no_beep_prob += no_beep_prob
        self.normalize_probs += norm_prob

class One_Crew_Search_DS(Crew_Search_Data):
    def __init__(self, ship):
        super(One_Crew_Search_DS, self).__init__()
        self.crew_cells = list(ship.open_cells) # list of all possible crew cells
        self.crew_cells.append(ship.crew_1)
        self.crew_cells.append(ship.crew_2)
        self.followup_list = list()

    def update_cell_mov_vals(self, crew_probs, curr_pos, cell_cord):
        crew_probs.bot_distance = get_manhattan_distance(curr_pos, cell_cord)
        crew_probs.beep_given_crew = LOOKUP_E[crew_probs.bot_distance]
        crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]
        crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
        self.beep_prob += crew_probs.crew_and_beep
        self.no_beep_prob += crew_probs.crew_and_no_beep

    def remove_cell_probs(self, rem_cell, curr_pos, logger):
        if rem_cell.cord not in self.crew_cells:
            return False

        if self.normalize_probs == 1:
            self.followup_list.clear()

        self.crew_cells.remove(rem_cell.cord)
        self.normalize_probs -= rem_cell.crew_probs.crew_prob
        self.followup_list.append((rem_cell.cord, rem_cell.crew_probs.crew_prob))
        logger.print(LOG_DEBUG, f"Removing cell {rem_cell.cord} from list of probable crew cells{self.crew_cells}")
        return True

    def init_crew_calcs(self, ship, curr_pos):
        crew_cell_size = len(self.crew_cells)
        self.set_all_probs()
        for cell_cord in self.crew_cells:
            cell = ship.get_cell(cell_cord)
            crew_probs = cell.crew_probs
            crew_probs.crew_prob = 1/crew_cell_size
            crew_probs.bot_distance = get_manhattan_distance(curr_pos, cell_cord)
            crew_probs.beep_given_crew = LOOKUP_E[crew_probs.bot_distance]
            crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]
            crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
            crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
            self.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

class Two_Crew_Search_DS(One_Crew_Search_DS):
    def __init__(self, ship):
        super(Two_Crew_Search_DS, self).__init__(ship)
        self.crew_cells_pair = dict()
        crew_cells_len = len(self.crew_cells)
        self.saved_crew_index = self.pending_crew_to_save = 0
        self.saved_crew_cell = ()
        self.crew_cells_length = crew_cells_len * (crew_cells_len - 1)

    def update_cell_mov_vals(self, crew_probs, curr_pos, cell_cord):
        crew_probs.bot_distance = get_manhattan_distance(curr_pos, cell_cord)
        crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]

    def update_crew_pair(self, cell_pair, curr_pos, pending_crew):
        cell_1 = cell_pair.cell_1.cord
        cell_2 = cell_pair.cell_2.cord
        cell_pair.cell_distances = 0
        if pending_crew == 1:
            self.update_cell_mov_vals(cell_pair.cell_1.crew_probs, curr_pos, cell_1)
            cell_pair.cell_2.crew_probs.bot_distance = 0
            cell_pair.cell_2.crew_probs.no_beep_given_crew = 1
            # print(1, curr_pos, cell_1, cell_pair.cell_1.crew_probs.bot_distance, cell_pair.cell_1.crew_probs.no_beep_given_crew, LOOKUP_NOT_E[cell_pair.cell_1.crew_probs.bot_distance])
        elif pending_crew == 2:
            cell_pair.cell_1.crew_probs.bot_distance = 0
            cell_pair.cell_1.crew_probs.no_beep_given_crew = 1
            self.update_cell_mov_vals(cell_pair.cell_2.crew_probs, curr_pos, cell_2)
            # print(2, curr_pos, cell_2, cell_pair.cell_2.crew_probs.bot_distance, cell_pair.cell_2.crew_probs.no_beep_given_crew, LOOKUP_NOT_E[cell_pair.cell_2.crew_probs.bot_distance])
        else:
            cell_pair.cell_distances = get_manhattan_distance(cell_1, cell_2)
            self.update_cell_mov_vals(cell_pair.cell_1.crew_probs, curr_pos, cell_1)
            self.update_cell_mov_vals(cell_pair.cell_2.crew_probs, curr_pos, cell_2)
            # print(0, curr_pos, cell_2, cell_pair.cell_2.crew_probs.bot_distance, cell_pair.cell_2.crew_probs.no_beep_given_crew, LOOKUP_NOT_E[cell_pair.cell_2.crew_probs.bot_distance])
            # print(0, curr_pos, cell_1, cell_pair.cell_1.crew_probs.bot_distance, cell_pair.cell_1.crew_probs.no_beep_given_crew, LOOKUP_NOT_E[cell_pair.cell_1.crew_probs.bot_distance])

    def update_cell_pair_vals(self, cell_pair, curr_pos, pending_crew = 0):
        self.update_crew_pair(cell_pair, curr_pos, pending_crew)
        crew_probs = cell_pair.crew_probs
        # print("Check update_cell_pair_vals...", cell_pair.cell_1.crew_probs.no_beep_given_crew, cell_pair.cell_2.crew_probs.no_beep_given_crew)
        crew_probs.beep_given_crew = 1 - cell_pair.cell_1.crew_probs.no_beep_given_crew * cell_pair.cell_2.crew_probs.no_beep_given_crew
        crew_probs.no_beep_given_crew = cell_pair.cell_1.crew_probs.no_beep_given_crew * cell_pair.cell_2.crew_probs.no_beep_given_crew
        crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
        # print("Check update_cell_pair_vals...", cell_pair.cell_1.cord, cell_pair.cell_2.cord, crew_probs.crew_prob, crew_probs.beep_given_crew, crew_probs.no_beep_given_crew, crew_probs.crew_and_beep, crew_probs.crew_and_no_beep)
        self.beep_prob += crew_probs.crew_and_beep
        self.no_beep_prob += crew_probs.crew_and_no_beep

    def print_map(self, cell):
        return

        # print(self.crew_cells, cell)
        # for key in self.crew_cells_pair:
        #     print(str(key) + "::", end="")
        #     length = len(self.crew_cells_pair[key])
        #     for itr, val in enumerate(self.crew_cells_pair[key]):
        #         if itr < length - 1:
        #             print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_and_beep) + ", ", end="")
        #         else:
        #             print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_and_beep), end="")
        #     print()

        for key in self.crew_cells_pair:
            print(str(key) + "::", end="")
            length = len(self.crew_cells_pair[key])
            for itr, val in enumerate(self.crew_cells_pair[key]):
                if itr < length - 1:
                    print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_prob) + ", ", end="")
                else:
                    print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_prob), end="")
            print()

        # for key in self.crew_cells_pair:
        #     print(str(key) + "::", end="")
        #     length = len(self.crew_cells_pair[key])
        #     for itr, val in enumerate(self.crew_cells_pair[key]):
        #         if itr < length - 1:
        #             print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_and_no_beep) + ", ", end="")
        #         else:
        #             print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_and_no_beep), end="")
        #     print()

    def update_rem_crew_probs(self, crew_pair):
        self.normalize_probs -= crew_pair.crew_probs.crew_prob
        self.crew_cells_length -= 1
        del crew_pair

    # invoke when only 1 crew member is present
    def remove_cell_probs_1(self, rem_cell, curr_pos):
        if curr_pos not in self.crew_cells:
            return

        index = self.crew_cells.index(rem_cell)
        if self.pending_crew_to_save == 1: # 1 is left, so remove key here
            cells_pair_list = self.crew_cells_pair[curr_pos]
            self.update_rem_crew_probs(cells_pair_list.pop(0))
            del self.crew_cells_pair[curr_pos]
        else: # 2 is left, so remove based on value here
            if (index > self.saved_crew_index):
                index -= 1
            else:
                self.saved_crew_index -= 1
            cell_pair_list = self.crew_cells_pair[self.saved_crew_cell]
            self.update_rem_crew_probs(cell_pair_list.pop(index))
            if not len(cell_pair_list):
                del self.crew_cells_pair[self.saved_crew_cell]

        self.crew_cells.pop(index)

    # invoke when 2 crew members are present
    def remove_cell_probs_2(self, rem_cell, curr_pos):
        index = self.crew_cells.index(rem_cell)
        for itr, key in enumerate(self.crew_cells):
            if itr < index:
                self.update_rem_crew_probs(self.crew_cells_pair[key].pop(index - 1))
            elif itr > index:
                self.update_rem_crew_probs(self.crew_cells_pair[key].pop(index))
            else:
                for crew_pair in self.crew_cells_pair[key]:
                    self.update_rem_crew_probs(crew_pair)

                del self.crew_cells_pair[key]

        self.crew_cells.remove(rem_cell)

    def remove_cell_probs(self, rem_cell, curr_pos, logger):
        if rem_cell.cord not in self.crew_cells:
            return False

        # logger.print(LOG_NONE, "self.is_change_rem_order", self.is_change_rem_order)
        if self.pending_crew_to_save:
            self.remove_cell_probs_1(rem_cell.cord, curr_pos)
        else:
            self.remove_cell_probs_2(rem_cell.cord, curr_pos)

        self.print_map(curr_pos)
        return True

    def retain_success_cell_probs(self, curr_pos, pending_crew, logger):
        if curr_pos not in self.crew_cells:
            return

        index = self.crew_cells.index(curr_pos)
        if pending_crew == 1: # retain cell 1, sinceccell 2 was found
            for itr, key in enumerate(self.crew_cells):
                final_index = index
                if itr < index:
                    final_index = index - 1
                elif itr == index:
                    cells_pair_list = self.crew_cells_pair[key]
                    for val_itr in range(len(self.crew_cells) - 1):
                        self.update_rem_crew_probs(cells_pair_list.pop(0))
                    del self.crew_cells_pair[key]
                    continue

                cells_pair_list = self.crew_cells_pair[key]
                for val_itr in range(len(self.crew_cells) - 1):
                    if val_itr < final_index:
                        self.update_rem_crew_probs(cells_pair_list.pop(0))
                    elif val_itr > final_index:
                        self.update_rem_crew_probs(cells_pair_list.pop(1))
        else: # retain cell 2, since cell 1 was found
            for itr, key in enumerate(self.crew_cells):
                if itr != index:
                    cells_pair_list = self.crew_cells_pair[key]
                    for val_itr in range(len(self.crew_cells) - 1):
                        self.update_rem_crew_probs(cells_pair_list.pop(0))
                    del self.crew_cells_pair[key]

        self.crew_cells.remove(curr_pos)
        self.pending_crew_to_save = pending_crew
        self.saved_crew_index = index
        self.saved_crew_cell = curr_pos
        logger.print(LOG_DEBUG, f"Retaining following cell {curr_pos} from the list of probable crew cells {self.crew_cells}")
        self.print_map(curr_pos)

    def init_crew_calcs(self, ship, curr_pos):
        self.set_beeps_prob()
        len_crew_cells = len(self.crew_cells)
        for key_itr, key in enumerate(self.crew_cells):
            self.crew_cells_pair[key] = list()
            cells_pair_list = self.crew_cells_pair[key]
            for val_itr, val in enumerate(self.crew_cells):
                if key_itr == val_itr:
                    continue
                cell_pair = Cell_Pair(key, val, ship)
                cell_pair.crew_probs.crew_prob = 1/self.crew_cells_length
                self.update_cell_pair_vals(cell_pair, curr_pos)
                cells_pair_list.append(cell_pair)

        self.print_map(curr_pos)


""" Core Ship Layout (same as last project) """

class Ship:
    def __init__(self, size, log_level = LOG_INFO):
        self.size = size
        self.grid = [[Cell(i, j, CLOSED_CELL) for j in range(size)] for i in range(size)] # grid is a list of list with each cell as a class
        self.open_cells = []
        self.logger = Logger(log_level)
        self.isBeep = 0
        self.bot = (0, 0)
        self.aliens = []
        self.initial_alien_pos = []
        self.crew_1 = (0, 0)
        self.crew_2 = (0, 0)

        self.generate_grid()

    def get_cell(self, cord):
        return self.grid[cord[0]][cord[1]]

    def generate_grid(self):
        self.assign_start_cell()
        self.unblock_closed_cells()
        self.unblock_dead_ends()
        self.compute_adj_cells()

    def compute_adj_cells(self):
        for cell_cord in self.open_cells:
            cell = self.get_cell(cell_cord)
            neighbors = get_neighbors(
                self.size,
                cell_cord,
                self.grid,
                OPEN_CELL
            )

            cell.adj_cells = neighbors

    def assign_start_cell(self):
        random_row = randint(0, self.size - 1)
        random_col = randint(0, self.size - 1)
        self.grid[random_row][random_col].cell_type = OPEN_CELL
        self.open_cells.append((random_row, random_col))

    def unblock_closed_cells(self):
        available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)
        while len(available_cells):
            closed_cell = choice(available_cells)
            self.get_cell(closed_cell).cell_type = OPEN_CELL
            self.open_cells.append(closed_cell)
            available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)

    def unblock_dead_ends(self):
        dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
        half_len = len(dead_end_cells)/2

        while half_len > 0:
            dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
            half_len -= 1
            if len(dead_end_cells):
                continue
            dead_end_cell = choice(dead_end_cells)
            closed_neighbors = get_neighbors(
                self.size, dead_end_cell, self.grid, CLOSED_CELL
            )
            random_cell = choice(closed_neighbors)
            self.get_cell(random_cell).cell_type = OPEN_CELL
            self.open_cells.append(random_cell)

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

    def set_player_cell_type(self):
        self.get_cell(self.bot).cell_type = BOT_CELL
        self.get_cell(self.crew_1).cell_type = CREW_CELL
        self.get_cell(self.crew_2).cell_type = CREW_CELL

    def place_crews(self):
        self.bot = choice(self.open_cells)
        self.open_cells.remove(self.bot)
        self.crew_1 = choice(self.open_cells)
        while (True):
            self.crew_2 = choice(self.open_cells)
            if self.crew_2 != self.crew_1:
                break

    def place_players(self):
        self.place_crews()
        self.open_cells.remove(self.crew_1)
        self.open_cells.remove(self.crew_2)
        for cell_cord in self.open_cells:
            self.init_cell_details(cell_cord)

        for cell in (self.crew_1, self.crew_2, self.bot):
            self.init_cell_details(cell)

        self.set_player_cell_type()
        self.reset_detection_zone()

    def init_cell_details(self, cell_cord):
        cell = self.get_cell(cell_cord)
        cell.listen_beep.crew_1_dist = get_manhattan_distance(cell_cord, self.crew_1)
        cell.listen_beep.crew_2_dist = get_manhattan_distance(cell_cord, self.crew_2)
        cell.listen_beep.c1_beep_prob = LOOKUP_E[cell.listen_beep.crew_1_dist]
        cell.listen_beep.c2_beep_prob = LOOKUP_E[cell.listen_beep.crew_2_dist]
        cell.cord = cell_cord

    # The following methods are called from the bot
    def place_aliens(self, no_of_aliens):
        pending_aliens = no_of_aliens
        cells_within_zone = self.cells_within_bot_zone
        all_cells = list(self.open_cells)
        all_cells.extend([self.crew_1, self.crew_2])
        self.initial_alien_cells = [cell_cord for cell_cord in all_cells if cell_cord not in cells_within_zone]
        while(pending_aliens > 0):
            alien_cell = choice(self.initial_alien_cells)
            if alien_cell in self.initial_alien_pos: continue
            self.initial_alien_pos.append(alien_cell)

            if self.get_cell(alien_cell).cell_type & CREW_CELL:
                self.get_cell(alien_cell).cell_type |= ALIEN_CELL
            else:
                self.get_cell(alien_cell).cell_type = ALIEN_CELL

            pending_aliens -= 1

        self.aliens = self.initial_alien_pos

    def move_aliens(self, bot):
        for itr, alien in enumerate(self.aliens):
            alien_cell = self.get_cell(alien)
            adj_cells = alien_cell.adj_cells
            alien_possible_moves = [adj_cell for adj_cell in adj_cells if self.get_cell(adj_cell).cell_type & ALIEN_MOVEMENT_CELLS]

            if len(alien_possible_moves) == 0:
                self.logger.print(
                    LOG_DEBUG,
                    f"Alien has no moves"
                )
                continue

            self.logger.print(
                LOG_DEBUG,
                f"Alien has moves {alien_possible_moves}"
            )

            alien_new_pos = choice(alien_possible_moves)
            old_alien_pos = alien_cell.cord
            self.aliens[itr] = alien_new_pos

            next_cell = self.get_cell(alien_new_pos)
            curr_cell = self.get_cell(old_alien_pos)

            if curr_cell.cell_type & CREW_CELL:
                curr_cell.cell_type = CREW_CELL
            else:
                curr_cell.cell_type = OPEN_CELL

            if next_cell.cell_type & BOT_CELL:
                self.logger.print(
                    LOG_DEBUG,
                    f"Alien moves from current cell {old_alien_pos} to bot cell {alien_new_pos}",
                )
                self.bot_caught_cell = alien_new_pos
                next_cell.cell_type = BOT_CAUGHT_CELL
                bot.is_caught = True
                return True

            else:
                self.logger.print(
                    LOG_DEBUG,
                    f"Alien moves from current cell {old_alien_pos} to open cell {alien_new_pos}",
                )
                if next_cell.cell_type & CREW_CELL:
                    next_cell.cell_type |= ALIEN_CELL
                else:
                    next_cell.cell_type = ALIEN_CELL

        return False

    def get_detection_zone(self, cell):
        k = ALIEN_ZONE_SIZE
        cells_within_zone = []
        min_row = max(0, cell[0] - k)
        max_row = min(self.size - 1, cell[0] + k)
        min_col = max(0, cell[1] - k)
        max_col = min(self.size - 1, cell[1] + k)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cells_within_zone.append((row, col))

        return cells_within_zone

    def reset_detection_zone(self, curr_cell = None):
        if curr_cell is None: curr_cell = self.bot
        self.cells_within_bot_zone = self.get_detection_zone(curr_cell)

        # Reset cells outside detection zone to false
        for i, cells in enumerate(self.grid):
            for j, cell in enumerate(cells):
                cell.within_detection_zone = ((i, j) in self.cells_within_bot_zone)

    def crew_beep_1(self, cell):
        self.isBeep = uniform(0, 1)
        return True if self.isBeep <= self.get_cell(cell).listen_beep.c1_beep_prob else False

    def crew_beep_2(self, cell):
        self.isBeep = uniform(0, 1)
        return True if self.isBeep <= self.get_cell(cell).listen_beep.c2_beep_prob else False

    def alien_beep(self):
        beep_heard = [alien_pos for alien_pos in self.aliens if self.get_cell(alien_pos).within_detection_zone]
        return len(beep_heard) > 0

    def reset_grid(self):
        # Need to make change here
        self.set_player_cell_type()
        self.reset_detection_zone()
        for cell in self.open_cells:
            self.get_cell(cell).cell_type = OPEN_CELL


""" Basic search algorithm, and the parent class for our bot """

class SearchAlgo:
    alien_config = ONE_ALIEN

    def __init__(self, ship, log_level):
        self.ship = ship
        self.curr_pos = ship.bot
        self.last_pos = ()
        self.logger = Logger(log_level)
        # self.max_prob = 0.0
        # self.temp_pred_crew_cells = dict()
        self.to_visit_list = []
        self.next_best_crew_cells = []
        # Working on few issues, will fix it ASAP
        self.disable_alien_calculation = True
        self.place_aliens_handler()

    def euclid_distance(self,point1, point2):
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def total_distance(self,points, path):
        total = 0
        for i in range(len(path) - 1):
            total += self.euclid_distance(points[path[i]], points[path[i + 1]])
        return total

    def shortest_route(self, points):
        points.insert(0, self.curr_pos)
        n = len(points)
        shortest_distance = float('inf')
        shortest_path = None
        for path in permutations(range(n)):
            dist = self.total_distance(points, path)
            if dist < shortest_distance:
                shortest_distance = dist
                shortest_path = list(path)
        next_point_index = shortest_path[1]
        shortest_path_points = [points[i] for i in shortest_path]
        return next_point_index, shortest_path_points

    def search_path(self, dest_cell, curr_pos = None, unsafe_cells = []):
        if curr_pos is None:
            curr_pos = self.curr_pos

        bfs_queue = []
        visited_cells = set()
        bfs_queue.append((curr_pos, [curr_pos]))

        while bfs_queue:
            current_cell, path_traversed = bfs_queue.pop(0)
            if current_cell == dest_cell:
                return path_traversed
            elif (current_cell in visited_cells):
                continue

            visited_cells.add(current_cell)
            neighbors = get_neighbors(self.ship.size, current_cell, self.ship.grid, BOT_MOVEMENT_CELLS)
            for neighbor in neighbors:
                if ((neighbor not in visited_cells) and
                    (neighbor not in unsafe_cells)):
                    bfs_queue.append((neighbor, path_traversed + [neighbor]))

        return [] #God forbid, this should never happen

    def place_aliens_handler(self):
        self.ship.place_aliens(self.alien_config)


""" Main parent class for all our bots, contain most of the common bot logic """

class ParentBot(SearchAlgo):
    def __init__(self, ship, log_level):
        super(ParentBot, self).__init__(ship, log_level)
        self.alien_evasion_data = One_Alien_Evasion_Data(ship) # to do only in one alien cases
        # self.alien_evasion_data = One_Alien_Evasion_Data(ship) # to do only in one alien cases
        self.temp_search_data = Crew_Search_Data()
        self.crew_search_data = One_Crew_Search_DS(ship)
        self.total_crew_to_save = 2
        self.traverse_path = []
        self.unsafe_cells = []
        self.pred_crew_cells = list()
        self.rescued_crew = self.is_recalc_probs = self.is_keep_moving = self.is_caught = False
        self.is_escape_strategy = self.made_move = False
        self.recalc_pred_cells = True
        self.is_bot_moved = False
        self.total_beep = self.track_beep = self.pending_crew = 0
        self.logger.print_grid(self.ship.grid)
        self.path_traversed = list()
        self.path_traversed.append(self.curr_pos)
        self.all_crews = [self.ship.crew_1, self.ship.crew_2]

    def rescue_info(self):
        init_1_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_1_dist
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_1}")
        init_2_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_2_dist
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_2}")
        return int(init_1_distance + self.ship.get_cell(self.ship.crew_1).listen_beep.crew_2_dist)

    def handle_single_crew_beep(self):
        self.crew_search_data.is_beep_recv = self.ship.crew_beep_1(self.curr_pos)
        self.logger.print(LOG_DEBUG, self.crew_search_data.is_beep_recv)

    def handle_two_crew_beep(self):
        beep_recv_2 = beep_recv_1 = False
        if self.pending_crew == 1:
            beep_recv_1 = self.ship.crew_beep_1(self.curr_pos)
        elif self.pending_crew == 2:
            beep_recv_2 = self.ship.crew_beep_2(self.curr_pos)
        else:
            beep_recv_1 = self.ship.crew_beep_1(self.curr_pos)
            beep_recv_2 = self.ship.crew_beep_2(self.curr_pos)

        self.crew_search_data.is_beep_recv = beep_recv_1 or beep_recv_2
        self.logger.print(LOG_DEBUG, beep_recv_1, beep_recv_2, beep_recv_1 or beep_recv_2)

    def handle_crew_beep(self):
        self.handle_single_crew_beep() if self.total_crew_to_save == 1 else self.handle_two_crew_beep()

        self.total_beep += 1
        if self.crew_search_data.is_beep_recv:
            self.track_beep += 1

    def handle_alien_beep(self):
        self.alien_evasion_data.is_beep_recv = self.ship.alien_beep()
        if self.alien_evasion_data.is_beep_recv:
            self.alien_evasion_data.beep_count += 1

    # def handle_resuce_opr(self, crew, crew_no, total_iter, idle_steps):
    #     distance_1 = get_manhattan_distance(crew, self.ship.bot)
    #     self.logger.print(LOG_INFO, f"Congrats, you found crew member {crew_no} who was initially {distance_1} steps away from you after {total_iter} steps. You moved {total_iter - idle_steps} steps, and waited for {idle_steps} steps")
    #     self.pending_crew = 2 if (crew_no % 2) else 1
    #     self.total_crew_count -= 1
    #     self.rescued_crew = True

    def calc_initial_search_data(self):
        self.crew_search_data.init_crew_calcs(self.ship, self.curr_pos)

    def find_traverse_path(self):
        prob_crew_cell = ()
        # Also compute if it is not in any neighbour zones
        if self.is_escape_strategy:
            self.traverse_path = self.find_escape_path()
            if len(self.traverse_path) == 0:
                # Sit and pray
                return False
        else:
            if len(self.pred_crew_cells) != 0:
                prob_crew_cell = self.pred_crew_cells.pop(0)
                self.traverse_path = self.search_path(prob_crew_cell, None, self.unsafe_cells)
                if len(self.traverse_path) == 0:
                    self.traverse_path = self.search_path(prob_crew_cell)
                self.logger.print(LOG_DEBUG, f"New path found, {self.traverse_path}. Pending cells to explore, {self.pred_crew_cells}")

            if len(self.traverse_path) == 0: # some edge case handling
                self.logger.print(LOG_DEBUG, f"Bot in {self.curr_pos} with crew cells {self.crew_search_data.crew_cells} and last prob_crew_cell was {prob_crew_cell}")
                self.logger.print(LOG_DEBUG, f"Bot in {self.curr_pos} has to find crew({self.total_crew_to_save}) {self.ship.crew_1, self.ship.crew_2} with pending crew {self.pending_crew}")
                self.logger.print(LOG_DEBUG, f"pred_crew_cells::{self.pred_crew_cells}")
                self.logger.print(LOG_DEBUG, f"path_traversed {self.path_traversed}")
                return False

        self.traverse_path.pop(0)

        return True

    def find_escape_path(self):
        escape_path = self.find_nearest_safe_cell()
        if len(escape_path) == 0:
            return []

        escape_path.pop(0)

        return escape_path

    def find_nearest_safe_cell(self):
        curr_cell = self.ship.get_cell(self.curr_pos)
        bot_adj_cells = curr_cell.adj_cells
        safe_cells = list()
        least_prob_cells = list()
        least_alien_cells = list()

        for cell_cord in bot_adj_cells:
            cell = self.ship.get_cell(cell_cord)
            safe_cell = True
            if cell.alien_probs.alien_prob == ALIEN_NOT_PRESENT:
                for neighbour in cell.adj_cells:
                    alien_prob_sum = 0
                    if neighbour in self.unsafe_cells:
                        safe_cell = False
                        neighbour_cell = self.ship.get_cell(neighbour)
                        alien_prob_sum += neighbour_cell.alien_probs.alien_prob

                least_prob_cells.append((cell_cord, alien_prob_sum))

                if safe_cell: safe_cells.append(cell_cord)
            
            least_alien_cells.append((cell_cord, cell.alien_probs.alien_prob))
            


        if len(safe_cells) > 0:
            next_cell = choice(safe_cells)
            print(f'In escape path:: curr_pos {self.curr_pos}, safe_cell: {next_cell}')
            return [self.curr_pos, next_cell]
        else:
            if len(least_prob_cells) > 0:
                next_move_cells = sorted(least_prob_cells, key=lambda x: x[1])
            else:
                next_move_cells = sorted(least_alien_cells, key=lambda x: x[1])
            print(f'In escape path:: curr_pos {self.curr_pos}, least_alien_cell: {next_move_cells[0][0]}')
            return [self.curr_pos, next_move_cells[0][0]]

    def print_prob_grid(self, plot = False):
        is_beep_recv = self.alien_evasion_data.is_beep_recv
        # curr_pos = self.curr_pos
        prob_grid = []
        prob_spread = list()
        grid = self.ship.grid
        for i, cells in enumerate(grid):
            prob_cell = []
            for j, cell in enumerate(cells):
                if cell.cell_type == CLOSED_CELL:
                    prob_cell.append(float('nan'))
                    print(f'[{i}][{j}]: C', end=' ')
                else:
                    print(f'[{i}][{j}]: {cell.alien_probs.alien_prob}', end=' ')
                    prob_cell.append(cell.alien_probs.alien_prob)
                    if not cell.alien_probs.alien_prob in prob_spread:
                        prob_spread.append(cell.alien_probs.alien_prob)

            prob_grid.append(prob_cell)
            print()

        prob_spread.sort()
        max_len = len(prob_spread) - 1
        # prob_grid[curr_pos[0]][curr_pos[1]] = 0
        if plot:
            pyplot.figure(figsize=(35,35))
            pyplot.colorbar(pyplot.imshow(prob_grid, vmin=prob_spread[0], vmax=prob_spread[max_len]))
            pyplot.title("Beep recv" if is_beep_recv else "Beep not recv")
            pyplot.show()

    def normalize_crew_handler(self, normalize_data, is_no_beep, crew_probs):
        if normalize_data:
            crew_probs.crew_prob = 1 / self.crew_search_data.normalize_probs
        else:
            crew_probs.crew_prob /= self.crew_search_data.normalize_probs

        if is_no_beep:
            crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
            crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
            self.crew_search_data.beep_prob += crew_probs.crew_and_beep
            self.crew_search_data.no_beep_prob += crew_probs.crew_and_no_beep

    # def check_nearby_crew(self):
    #     if self.crew_search_data.is_beep_recv:
    #         return False # no need to check nearby cells when we get a beep

    #     neighbors = get_neighbors(self.ship.size, self.curr_pos, self.ship.grid, BOT_MOVEMENT_CELLS)
    #     neighbors_in_crew = [neighbor for neighbor in neighbors if neighbor in self.crew_search_data.crew_cells]
    #     if (len(neighbors_in_crew)):
    #         for neighbor in neighbors_in_crew:
    #             self.crew_search_data.remove_cell_probs(self.ship.get_cell(neighbor), self.curr_pos, self.logger)
    #             if neighbor in self.pred_crew_cells:
    #                 self.pred_crew_cells.remove(neighbor)

    #             if self.is_keep_moving:
    #                 self.update_move_vals()
    #
    #         self.logger.print(LOG_DEBUG, f"Following cells {neighbors_in_crew}, were removed from crew cells {self.crew_search_data.crew_cells} and pred cells {self.pred_crew_cells}")
    #         return True
    #     return False

    '''
        If beep heard, cells within detection zone: P(B obs /A) = 1
        cells outside P(B obs /A) = 0
        If beep not heard, cells within detection zone: P(B obs /A) = 0
        cells outside P(B obs /A) = 1
        P(A/B obs) = P(B obs/ A) P(A) / P(B obs)
    '''
    def update_alien_data(self):
        if (not self.alien_evasion_data.is_beep_recv) or self.made_move:
            self.alien_evasion_data.reset_alien_calcs()
    
        else:
            alien_cells = self.alien_evasion_data.alien_cells
            alien_cell_list = [cell_cord for cell_cord in alien_cells]
            beep_recv = self.alien_evasion_data.is_beep_recv
            self.alien_evasion_data.beep_prob = 0
            prob_cell_list = list()
            self.alien_evasion_data.present_alien_cells = []

            prob_alien_in_inner_cells = ALIEN_PRESENT if beep_recv else ALIEN_NOT_PRESENT
            prob_alien_in_outer_cells = ALIEN_PRESENT if (not beep_recv) else ALIEN_NOT_PRESENT
            total_unsafe_cells = self.alien_config * TOTAL_UNSAFE_CELLS

            # Likelihood computation
            for cell_cord in alien_cell_list:
                cell = self.ship.get_cell(cell_cord)
                if cell.alien_probs.alien_prob == ALIEN_NOT_PRESENT or (cell.cord == self.curr_pos):
                    cell.alien_probs.alien_prob = ALIEN_NOT_PRESENT
                    continue

                prob_beep_gv_alien = prob_alien_in_inner_cells if cell.within_detection_zone else prob_alien_in_outer_cells
                alien_prob = cell.alien_probs.alien_prob
                cell.alien_probs.alien_and_beep = prob_beep_gv_alien * alien_prob
                self.alien_evasion_data.beep_prob += cell.alien_probs.alien_and_beep
                self.alien_evasion_data.present_alien_cells.append(cell)

            # Updating the alien prob from prior knowledge
            for cell in self.alien_evasion_data.present_alien_cells:
                if cell.alien_probs.alien_and_beep == ALIEN_NOT_PRESENT:
                    cell.alien_probs.alien_prob = ALIEN_NOT_PRESENT
                    self.alien_evasion_data.present_alien_cells.remove(cell)
                    continue

                cell.alien_probs.alien_prob = cell.alien_probs.alien_and_beep/self.alien_evasion_data.beep_prob
                prob = cell.alien_probs.alien_prob
                # Set the top most likely alien cells
                if beep_recv:
                    prob_cell_list.append((prob, cell.cord))

                else:
                    if (len(prob_cell_list) < total_unsafe_cells):
                        prob_cell_list.append((prob, cell.cord))
                        prob_cell_list = sorted(prob_cell_list, key=lambda x: x[0], reverse=True)
                    elif prob > prob_cell_list[-1][0]:
                        prob_cell_list.remove(prob_cell_list[-1])
                        prob_cell_list.append((prob, cell.cord))
                        prob_cell_list = sorted(prob_cell_list, key=lambda x: x[0], reverse=True)

            self.unsafe_cells = [prob_cell[1] for prob_cell in prob_cell_list]

    def alien_beep(self):
        alien_cell = self.ship.get_cell(self.ship.alien)
        self.alien_evasion_data.is_beep_recv = alien_cell.within_detection_zone
        self.logger.print(
            LOG_DEBUG,
            f"alien_pos:{self.ship.alien}, within_detection_zone::{alien_cell.within_detection_zone}"
        )

    def compute_likely_alien_movements(self):
        if len(self.alien_evasion_data.present_alien_cells) == 0:
            self.alien_evasion_data.present_alien_cells = [self.ship.get_cell(cell_cord) for cell_cord in self.alien_evasion_data.alien_cells]

        prob_cell_mapping = dict()

        for cell in self.alien_evasion_data.present_alien_cells:
            self.logger.print(
                LOG_DEBUG, f"Iterating for ::{cell.cord}"
            )
            possible_moves = cell.adj_cells
            total_moves = len(possible_moves)

            if ((cell.alien_probs.alien_prob == ALIEN_NOT_PRESENT) or
                (total_moves == 0)):
                continue

            if cell.cord not in prob_cell_mapping.keys():
                prob_cell_mapping[cell.cord] = cell.alien_probs.alien_prob
                cell.alien_probs.alien_prob = 0

            self.logger.print(
                LOG_DEBUG, f"prob_cell_mapping::{prob_cell_mapping}"
            )

            self.logger.print(
                LOG_DEBUG, f"Neighbours for the current cell::{possible_moves}"
            )

            for alien_move in possible_moves:
                adj_cell = self.ship.get_cell(alien_move)
                if alien_move not in prob_cell_mapping.keys():
                    prob_cell_mapping[alien_move] = adj_cell.alien_probs.alien_prob
                    adj_cell.alien_probs.alien_prob = 0
                adj_cell.alien_probs.alien_prob += prob_cell_mapping[cell.cord]/total_moves

        prob_cell_mapping.clear()


    """
        Ideally it is better to move the bot in the direction of the highest prob
        To do this, pred_crew_cells should be sorted based on probabilty
        Remember, we are not taking into account where the alien will be here!!
    """
    def move_bot(self):
        if not len(self.traverse_path):
            return False

        self.ship.get_cell(self.curr_pos).cell_type = OPEN_CELL
        self.last_pos = self.curr_pos
        self.curr_pos = self.traverse_path.pop(0)
        curr_cell = self.ship.get_cell(self.curr_pos)
        if (curr_cell.cell_type & ALIEN_CELL):    #  OOPS BYE BYE
            curr_cell.cell_type |= ALIEN_CELL
            self.logger.print(LOG_INFO, f"Bot caught!!!! @ cell::{self.curr_pos}")
            self.is_caught = True
        elif (curr_cell.cell_type & CREW_CELL):
            curr_cell.cell_type |= BOT_CELL
            self.logger.print(LOG_INFO, f"Yay bot found a crew!! @ cell::{self.curr_pos}")
            self.made_move = False
        else:
            curr_cell.cell_type = BOT_CELL
            self.ship.reset_detection_zone(self.curr_pos)
            self.made_move = True

        self.logger.print(LOG_DEBUG, f"Bot {self.last_pos} has moved to {self.curr_pos} trying to find {self.total_crew_to_save} crew pending")
        return True

    def is_rescued(self):
        for itr, crew in enumerate(self.all_crews):
            if crew == self.curr_pos:
                if itr == 0:
                    self.pending_crew = 2
                else:
                    self.pending_crew = 1
                self.logger.print_all_crew_data(LOG_DEBUG, self)
                self.all_crews.remove(self.curr_pos)
                break

        if len(self.all_crews) == 0:
            return True

        return False

    def remove_cell(self, rem_cell):
        return

    def update_crew_search_data(self):
        return

    def calculate_best_path(self):
        if len(self.traverse_path):
            self.traverse_path = self.search_path(self.traverse_path[-1], None, self.unsafe_cells)

        if self.alien_evasion_data.is_beep_recv and len(self.traverse_path) == 0:
            self.traverse_path = self.find_escape_path() # Change this
            if len(self.traverse_path) == 0:
                # Sit and pray
                return False
        else:
            if len(self.traverse_path):
                return False

            prob_crew_cell = choice(self.next_best_crew_cells)
            self.traverse_path = self.search_path(prob_crew_cell)
            self.traverse_path.pop(0)
            self.logger.print(LOG_DEBUG, f"New path to cell {prob_crew_cell} was found, {self.traverse_path}")

        return True

    def start_rescue(self): # working, finalllly ;-;
        total_moves = idle_steps = total_iter = 0
        keep_moving = False
        init_distance = self.rescue_info()
        self.calc_initial_search_data()
        self.logger.print_all_crew_data(LOG_DEBUG, self)

        while (True): # Keep trying till you find the crew
            print(total_iter)
            if total_iter >= 1000:
                return init_distance, total_iter, total_moves, BOT_STUCK

            total_iter += 1
            idle_steps += 1
            self.handle_alien_beep()
            self.handle_crew_beep()

            if self.alien_evasion_data.beep_count > 0:
                self.compute_likely_alien_movements()

            self.update_alien_data()
            self.update_crew_search_data()

            if idle_steps == IDLE_BEEP_COUNT or self.alien_evasion_data.is_beep_recv:
                idle_steps = 0
                keep_moving = self.calculate_best_path()

            if keep_moving:
                if self.move_bot():
                    if self.is_rescued():
                        return init_distance, total_iter, total_moves, BOT_SUCCESS

                    elif self.is_caught:
                        return init_distance, total_iter, total_moves, BOT_FAILED

                    self.is_bot_moved = True
                    total_moves += 1
                    idle_steps = 0

                    if self.curr_pos in self.crew_search_data.crew_cells:
                        self.remove_cell(self.curr_pos)
                else:
                    idle_steps += 1
                    keep_moving = False
                    self.made_move = False

            if self.ship.move_aliens(self):
                return init_distance, total_iter, total_moves, BOT_FAILED

    def max_alien_beep(self):
        k = ALIEN_ZONE_SIZE
        if k == 1:
            return 0
        else:
            return ceil(k/2)

""" Bot 1 as per given specification """
class Bot_1(ParentBot):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_1, self).__init__(ship, log_level)
        self.override_ship_details()

    def rescue_info(self):
        if self.total_crew_to_save == 1:
            init_1_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_1_dist
            self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_1}")
            return int(init_1_distance)
        else:
            return super(Bot_1, self).rescue_info()

    def override_ship_details(self):
        self.total_crew_to_save = self.total_crew_count = self.pending_crew = 1
        cell = self.ship.get_cell(self.ship.crew_2)
        if cell.cell_type & ALIEN_CELL:
            self.ship.get_cell(self.ship.crew_2).cell_type = ALIEN_CELL
        else:
            self.ship.get_cell(self.ship.crew_2).cell_type = OPEN_CELL

    def init_alien_calcs(self):
        alien_cell_size = len(self.alien_evasion_data.init_alien_cells)
        for cell_cord in self.alien_evasion_data.init_alien_cells:
            cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
            cell.alien_probs.alien_prob = 1/alien_cell_size

    def remove_cell(self, rem_cell):
        crew_search_data = self.crew_search_data
        if rem_cell not in crew_search_data.crew_cells:
            return

        crew_probs = self.ship.get_cell(rem_cell).crew_probs
        crew_search_data.normalize_probs -= crew_probs.crew_prob
        crew_probs.crew_prob = 0 # can't use this as our if condition because the crew_probs tend to 0 sometimes
        crew_search_data.crew_cells.remove(rem_cell)
        self.logger.print(LOG_DEBUG, f"Removed {rem_cell} cell from possible crew cells {crew_search_data.crew_cells}")

    def remove_nearby_cells(self):
        crew_search_data = self.crew_search_data
        neighbors = get_neighbors(self.ship.size, self.curr_pos, self.ship.grid, BOT_MOVEMENT_CELLS)
        for neighbor in neighbors:
            self.remove_cell(neighbor)

    def norm_probs(self):
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()
        alpha_num = 0.0
        if crew_search_data.normalize_probs <= 0:
            crew_search_data.normalize_probs = 0
            self.logger.print_all_crew_data(LOG_DEBUG, self)
            self.logger.print(LOG_DEBUG, "Using additive smoothing...")
            alpha_num = ADDITIVE_VALUE
            crew_search_data.normalize_probs += alpha_num * len(crew_search_data.crew_cells)
        elif crew_search_data.normalize_probs > 1.5:
            self.logger.print_all_crew_data(LOG_NONE, self)
            print("THERE WAS A MAJOR NEWS!!!!")
            exit(0)

        for cell_cord in crew_search_data.crew_cells:
            cell = self.ship.get_cell(cell_cord)
            crew_probs = cell.crew_probs
            if alpha_num:
                crew_probs.crew_prob = crew_probs.crew_prob + alpha_num

            crew_probs.crew_prob = crew_probs.crew_prob / crew_search_data.normalize_probs
            if (self.is_bot_moved):
                crew_probs.bot_distance = get_manhattan_distance(self.curr_pos, cell_cord)
                crew_probs.beep_given_crew = LOOKUP_E[crew_probs.bot_distance]
                crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]
            crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
            crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
            temp_search_data.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def update_cell_crew_probs(self, crew_probs):
        crew_probs.crew_given_beep = crew_probs.crew_and_beep/self.crew_search_data.beep_prob
        if self.crew_search_data.no_beep_prob: # it is possible to never not hear beeps depending on how far we have searched. can norm if required.
            crew_probs.crew_given_no_beep = crew_probs.crew_and_no_beep/self.crew_search_data.no_beep_prob
        crew_probs.crew_prob = crew_probs.crew_given_beep if self.crew_search_data.is_beep_recv else crew_probs.crew_given_no_beep
        crew_probs.crew_and_beep = crew_probs.beep_given_crew*crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew*crew_probs.crew_prob
        self.temp_search_data.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

    def update_all_crew_probs(self):
        crew_search_data = self.crew_search_data
        if not crew_search_data.is_beep_recv: # no beep heard
            self.remove_nearby_cells()

        if self.is_bot_moved or (crew_search_data.normalize_probs < 1): # normalize our probs
            self.norm_probs()
            self.is_bot_moved = False

        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def update_crew_search_data(self):
        self.update_all_crew_probs()

        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        self.next_best_crew_cells.clear()
        temp_search_data.set_all_probs()
        max_prob = 0

        for cell_cord in crew_search_data.crew_cells: # update probs for this round
            cell = self.ship.get_cell(cell_cord)
            crew_probs = cell.crew_probs
            self.update_cell_crew_probs(crew_probs)

            if crew_probs.crew_prob > max_prob:
                self.next_best_crew_cells.clear()
                max_prob = crew_probs.crew_prob

            if max_prob == crew_probs.crew_prob:
                self.next_best_crew_cells.append(cell_cord)

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def is_rescued(self): # since this will only be used by 1, 2 and 3 (reason for if condition), this has been overriden
        if self.total_crew_count == 1:
            if self.curr_pos == self.ship.crew_1:
                self.logger.print(LOG_INFO, f"Bot has saved crew member at cell {self.ship.crew_1}")
                return True
            return False
        else:
            return super(Bot_1, self).is_rescued()


class Bot_3(Bot_1):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_3, self).__init__(ship, log_level)
        self.override_ship_details()

    def is_rescued(self):
        old_val = self.pending_crew
        ret_val = super(Bot_3, self).is_rescued()
        if old_val != self.pending_crew:
            self.calc_initial_search_data()
            self.is_bot_moved = False # recalc already, need not do it again...

        return ret_val

    def override_ship_details(self):
        self.total_crew_to_save = self.total_crew_count = 2
        self.pending_crew = 0
        cell = self.ship.get_cell(self.ship.crew_2)
        if cell.cell_type & ALIEN_CELL:
            self.ship.get_cell(self.ship.crew_2).cell_type |= CREW_CELL
        else:
            self.ship.get_cell(self.ship.crew_2).cell_type = CREW_CELL

class Bot_4(ParentBot):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_4, self).__init__(ship, log_level)
        self.crew_search_data = Two_Crew_Search_DS(ship)

    def remove_nearby_cells(self):
        return

    def norm_probs(self):
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()
        alpha_num = 0.0
        print_after = False
        if crew_search_data.normalize_probs <= 0:
            print("")
        elif crew_search_data.normalize_probs > 1.5:
            self.logger.print_all_crew_data(LOG_NONE, self)
            print("THERE WAS A MAJOR NEWS!!!!")
            exit(0)


    def update_all_crew_probs(self):
        crew_search_data = self.crew_search_data
        if not crew_search_data.is_beep_recv: # no beep heard
            self.remove_nearby_cells()

        if self.is_bot_moved or (crew_search_data.normalize_probs < 1): # normalize our probs
            self.norm_probs()
            self.is_bot_moved = False

        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def update_crew_search_data(self):
        self.update_all_crew_probs()

        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data


#     def update_pred_crew_cells(self):
#         self.pred_crew_cells.clear()
#         cell_pair = None
#         max_prob_cells = self.temp_pred_crew_cells[self.max_prob]
#         curr_pos_beeps = self.ship.get_cell(self.curr_pos).crew_probs.track_beep
#         curr_pos_fq = curr_pos_beeps[0]/curr_pos_beeps[1] if curr_pos_beeps[1] else 0
#         if len(self.last_pos) and curr_pos_fq:
#             last_pos_beeps = self.ship.get_cell(self.last_pos).crew_probs.track_beep
#             last_pos_fq = last_pos_beeps[0]/last_pos_beeps[1] if last_pos_beeps[1] else 0
#             max_prob_cells = sorted(max_prob_cells, key=lambda cell_pair: cell_pair[1], reverse=True)
#             if curr_pos_fq > last_pos_fq:
#                 cell_pair = max_prob_cells[-1][0]
#             elif last_pos_fq > curr_pos_fq:
#                 cell_pair = max_prob_cells[0][0]
#             else:
#                 pos = round(len(max_prob_cells)/2)
#                 cell_pair = max_prob_cells[pos][0]
#         else: # can be randomized now...
#             cell_pair = choice(max_prob_cells)[0]

#         distance_1 = cell_pair.cell_1.crew_probs.bot_distance + cell_pair.cell_distances
#         distance_2 = cell_pair.cell_2.crew_probs.bot_distance + cell_pair.cell_distances
#         if distance_1 > distance_2:
#             if cell_pair.cell_2.cord != self.curr_pos:
#                 self.pred_crew_cells.append(cell_pair.cell_2.cord)
#             self.pred_crew_cells.append(cell_pair.cell_1.cord)
#         else:
#             if cell_pair.cell_1.cord != self.curr_pos:
#                 self.pred_crew_cells.append(cell_pair.cell_1.cord)
#             self.pred_crew_cells.append(cell_pair.cell_2.cord)

#         self.logger.print(LOG_DEBUG, f"The new pred crew cells are {self.pred_crew_cells}")
#         self.temp_pred_crew_cells.clear()
#         self.max_prob = 0.0

#     def normalize_probs(self, is_no_beep):
#         normalize_data = False
#         if is_no_beep or self.crew_search_data.normalize_probs == 0:
#             self.crew_search_data.normalize_probs = self.crew_search_data.crew_cells_length
#             normalize_data = True

#         for key in self.crew_search_data.crew_cells_pair:
#             cell_pair_list = self.crew_search_data.crew_cells_pair[key]
#             for cell_pair in cell_pair_list:
#                 self.normalize_crew_handler(normalize_data, is_no_beep, cell_pair.crew_probs)

#         self.crew_search_data.print_map(self.curr_pos)

#     def update_all_crew_probs(self):
#         for key in self.crew_search_data.crew_cells_pair:
#             cell_pair_list = self.crew_search_data.crew_cells_pair[key]
#             for cell_pair in cell_pair_list:
#                 self.crew_search_data.update_cell_pair_vals(cell_pair, self.curr_pos, self.pending_crew)

#     def update_crew_search_data(self):
#         beep_prob = no_beep_prob = 0.0
#         self.logger.print(LOG_DEBUG, f"is_beep_recv::{self.crew_search_data.is_beep_recv}, {self.ship.isBeep}")
#         self.start_crew_search_data()
#         check_for_failure(self, 3)
#         for key in self.crew_search_data.crew_cells_pair:
#             cell_pair_list = self.crew_search_data.crew_cells_pair[key]
#             for cell_pair in cell_pair_list:
#                 crew_probs = cell_pair.crew_probs
#                 if self.recalc_pred_cells:
#                     distance_1 = cell_pair.cell_1.crew_probs.bot_distance + cell_pair.cell_distances
#                     distance_2 = cell_pair.cell_2.crew_probs.bot_distance + cell_pair.cell_distances
#                     crew_probs.bot_distance = distance_1 if distance_1 < distance_2 else distance_2
#                 self.handle_crew_search_data(crew_probs, cell_pair)
#                 beep_prob += crew_probs.crew_and_beep
#                 no_beep_prob += crew_probs.crew_and_no_beep

#         self.logger.print(LOG_DEBUG, self.crew_search_data.normalize_probs)
#         self.crew_search_data.print_map(self.curr_pos)
#         self.crew_search_data.set_beeps_prob(beep_prob, no_beep_prob)
#         self.end_crew_search_data()

#     def update_crew_cells(self):
#         if self.rescued_crew:
#             self.crew_search_data.retain_success_cell_probs(self.curr_pos, self.pending_crew, self.logger)
#             self.rescued_crew = False
#             self.is_recalc_probs = True
#         else:
#             super(Bot_4, self).update_crew_cells()

#     # repeat, to be shifted to a common class!!
#     def start_rescue(self):
#         is_move_bot = False
#         beep_counter = idle_steps = total_iter = 0
#         init_distance = self.rescue_info()

#         self.calc_initial_search_data()
#         while(True):
#             # self.ship.logger.print_grid(self.ship.grid)
#             # if (total_iter > 50):
#             #     exit()
#             total_iter += 1
#             self.handle_alien_beep()
#             self.handle_crew_beep()
#             self.update_alien_data()
#             self.update_crew_search_data()
#             if self.recalc_pred_cells:
#                 self.update_pred_crew_cells()

#             if (self.is_keep_moving or is_move_bot) and self.move_bot():
#                 if self.is_rescued(total_iter, idle_steps):
#                     return (init_distance, total_iter, idle_steps)

#                 elif self.is_caught:
#                     return (total_iter, self.curr_pos)

#                 self.update_crew_cells()
#                 beep_counter = 0
#                 is_move_bot = False
#                 self.ship.reset_detection_zone(self.curr_pos)
#             else:
#                 is_move_bot = False
#                 idle_steps += 1

#             # if self.ship.move_aliens(self) and self.is_caught:
#             #     return (total_iter, self.curr_pos)
#             # update probability of alien movement based on current P(A), cell.alien_prob
#             # self.compute_likely_alien_movements()

#             beep_counter += 1
#             if beep_counter >= (IDLE_BEEP_COUNT - 1):
#                 is_move_bot = True

"""Simulation & Testing logic begins"""

BOT_NAMES = {
    0 : "bot_1",
    1 : "bot_3",
    2 : "bot_4"
}

# Responsible for updating the alpha for each worker pool
def update_lookup(alpha):
    global LOOKUP_E, LOOKUP_NOT_E, ALPHA
    ALPHA = alpha
    LOOKUP_E = [(pow(exp, (-1*ALPHA*(i - 1)))) for i in range(GRID_SIZE*2 + 1)]
    LOOKUP_NOT_E = [(1-LOOKUP_E[i]) for i in range(GRID_SIZE*2 + 1)]

def bot_factory(itr, ship, log_level = LOG_NONE):
    if (itr == 0):
        return Bot_1(ship, log_level)
    if (itr == 1):
        return Bot_3(ship, log_level)
    # elif (itr == 2):
    #     return Bot_4(ship, log_level)
    return ParentBot(ship, log_level)

# Test function
def run_test(log_level = LOG_INFO):
    update_lookup(ALPHA)
    for itr in range(1):
        ship = Ship(GRID_SIZE, log_level)
        ship.place_players()
        for i in range(TOTAL_BOTS):
            print(BOT_NAMES[i], i)
            begin = time()
            bot = bot_factory(i, ship, log_level)
            print(bot.start_rescue())
            end = time()
            print(end - begin)
            del bot
            ship.reset_grid()
        del ship

class FINAL_OUT:
    def __init__(self) -> None:
        self.distance = 0
        self.total_iter = 0
        self.total_moves = 0
        self.idle_moves = 0
        self.success = 0
        self.failure = 0
        self.stuck = 0
        self.time_taken = 0.0
        pass

# Runs n number of iteration for each bot for given alpha value
def run_sim(iterations_range, queue, alpha_range):
    alpha_dict = dict()
    for alpha in alpha_range:
        update_lookup(alpha)
        temp_data_set = [FINAL_OUT() for j in range(TOTAL_BOTS)]
        for itr in iterations_range:
            # print(itr+1, end = '\r')
            ship = Ship(GRID_SIZE)
            ship.place_players()
            # ship.logger.print_grid(ship.grid)
            for bot_no in range(TOTAL_BOTS):
                bot = bot_factory(bot_no, ship)
                begin = time()
                ret_vals = bot.start_rescue()
                end = time()
                temp_data_set[bot_no].distance += ret_vals[0]
                temp_data_set[bot_no].total_iter += ret_vals[1]
                temp_data_set[bot_no].total_moves += ret_vals[2]
                if ret_vals[3] == BOT_SUCCESS:
                    temp_data_set[bot_no].success += 1
                elif ret_vals[3] == BOT_FAILED:
                    temp_data_set[bot_no].failure += 1
                else:
                    temp_data_set[bot_no].stuck += 1
                temp_data_set[bot_no].time_taken += (end-begin)
                ship.reset_grid()
                del bot
            del ship
        alpha_dict[alpha] = temp_data_set
    queue.put(alpha_dict)

# Creates "n" process, and runs multiple simulation for same value of alpha simulataenously
def run_multi_sim(alpha_range, is_print = False):
    begin = time()
    alpha_dict = dict()
    data_set = [FINAL_OUT() for j in range(TOTAL_BOTS)]
    processes = []
    queue = Queue()
    if (is_print):
        print(f"Iterations begin...")
    core_count = cpu_count()
    total_iters = round(TOTAL_ITERATIONS/core_count)
    actual_iters = total_iters * core_count
    for itr in range(core_count):
        p = Process(target=run_sim, args=(range(itr*total_iters, (itr+1)*total_iters), queue, alpha_range))
        processes.append(p)
        p.start()

    for proc in processes:
        proc.join()
        temp_alpha_dict = queue.get()
        for alpha, value in temp_alpha_dict.items():
            if alpha not in alpha_dict:
                alpha_dict[alpha] = value
            else:
                for i, val_range in enumerate(value):
                    alpha_dict[alpha][i].distance += value[i].distance
                    alpha_dict[alpha][i].total_iter += value[i].total_iter
                    alpha_dict[alpha][i].total_moves += value[i].total_moves
                    alpha_dict[alpha][i].success += value[i].success
                    alpha_dict[alpha][i].failure += value[i].failure
                    alpha_dict[alpha][i].stuck += value[i].stuck
                    alpha_dict[alpha][i].time_taken += value[i].time_taken

    for alpha, resc_val in alpha_dict.items():
        for itr, value in enumerate(resc_val):
            value.distance /= actual_iters
            value.total_iter /= actual_iters
            value.total_moves /= actual_iters
            value.idle_moves = value.total_iter - value.total_moves
            value.success /= actual_iters
            value.failure /= actual_iters
            value.stuck /= actual_iters
            value.time_taken /= actual_iters
    end = time()

    if (is_print):
        for alpha, resc_val in alpha_dict.items():
            print()
            print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {alpha}")
            print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s" % ("Bot", "Success Rate", "Failure Rate", "Stuck", "Distance", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
            for itr, value in enumerate(resc_val):
                print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s" % (BOT_NAMES[itr], value.success, value.failure, value.stuck, value.distance, value.total_iter, value.idle_moves, value.total_moves, value.time_taken))
    else:
        print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took total time {end-begin} for alpha range {alpha_dict.keys()}")

    del queue
    del processes
    return alpha_dict

# Runs multiple simulations for multiple values of alpha concurrently
def compare_multiple_alpha():
    global ALPHA
    alpha_range = [round(ALPHA + (ALPHA_STEP_INCREASE * i), 2) for i in range(MAX_ALPHA_ITERATIONS)]
    alpha_dict = run_multi_sim(alpha_range)
    print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s" % ("Bot", "Success Rate", "Failure Rate", "Stuck", "Distance", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
    for alpha, resc_val in alpha_dict.items():
        print(f"{'*'*82}{alpha}{'*'*82}")
        for itr, value in enumerate(resc_val):
            print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s" % (BOT_NAMES[itr], value.success, value.failure, value.stuck, value.distance, value.total_iter, value.idle_moves, value.total_moves, value.time_taken))

# MAJOR ISSUES WITH ALL BOTS!!
if __name__ == '__main__':
    run_test()
    # run_multi_sim([ALPHA], True)
    # compare_multiple_alpha()
