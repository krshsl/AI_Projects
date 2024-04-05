from random import randint, uniform, choice
from math import e as exp, ceil, sqrt, floor
from inspect import currentframe
from time import time
from multiprocessing import Pool, cpu_count
from itertools import permutations
from heapq import heappop, heappush

#Constants
CLOSED_CELL = 1
OPEN_CELL = 2
BOT_CELL = 4
CREW_CELL = 8
ALIEN_CELL = 16
BOT_CAUGHT_CELL = 32
BOT_MOVEMENT_CELLS = OPEN_CELL | CREW_CELL | ALIEN_CELL
ALIEN_MOVEMENT_CELLS = CREW_CELL | OPEN_CELL | BOT_CELL
GRID_SIZE = 35
ADDITIVE_VALUE = 1e-6

BOT_SUCCESS = 1
BOT_FAILED = 2
BOT_STUCK = 3

X_COORDINATE_SHIFT = [1, 0, 0, -1]
Y_COORDINATE_SHIFT = [0, 1, -1, 0]

ALIEN_ZONE_SIZE = 5 # k - k >= 1, need to determine the large value
SEARCH_ZONE_SIZE = 5
ALPHA = 0.25 # avoid large alpha at the cost of performance
IDLE_BEEP_COUNT = 0
TOTAL_UNSAFE_CELLS = 6

TOTAL_ITERATIONS = 64
MAX_ALPHA_ITERATIONS = 10
MAX_K_ITERATIONS = 7
ALPHA_STEP_INCREASE = 0.05
ALIEN_ZONE_INCREASE = 1
TOTAL_BOTS = 8

DISTANCE_UTILITY=0.05 #DON'T WANT DISTANCE TO BE A MAJOR CONTRIBUTOR....
ALIEN_UTILITY=-1.5 #ALIENS ARE ALWAYS DANGEROUS!?!
CREW_UTILITY=1 #LET THIS PLAY SOME ROLE???

LOG_NONE = 0
LOG_DEBUG_ALIEN = 0.5
LOG_INFO = 1
LOG_DEBUG = 2
LOG_DEBUG_GRID = 3
IGNORE_GRID_DEBUG = True

LOOKUP_E = []
LOOKUP_NOT_E = []

ALIEN_NOT_PRESENT = 0.0
ALIEN_PRESENT = 1.0

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

    def print_all_crew_probs(self, curr_pos, ship, crew_search_data, total_beep, is_beep_recv):
        print(currentframe().f_back.f_back.f_code.co_name, "::", currentframe().f_back.f_back.f_lineno, curr_pos, total_beep, is_beep_recv)
        total_prob = 0.0
        for key in crew_search_data.crew_cells_pair:
            print(key, "::", end="\t")
            for cells in crew_search_data.crew_cells_pair[key]:
                total_prob += cells.crew_probs.crew_prob
                print(cells.cell_2.cord, "::", cells.crew_probs.crew_prob, end = " ")
            print()
        for key in crew_search_data.crew_cells_pair:
            print(key, "::", end="\t")
            for cells in crew_search_data.crew_cells_pair[key]:
                print(cells.cell_2.cord, "::", cells.crew_probs.no_beep_given_crew, end = " ")
            print()
        print(currentframe().f_back.f_back.f_code.co_name, "::", currentframe().f_back.f_back.f_lineno, curr_pos, total_prob, crew_search_data.crew_cells)

    def print_all_pair_data(self, log_level, bot):
        if not self.check_log_level(log_level):
            return

        self.print_all_crew_probs(bot.curr_pos, bot.ship, bot.crew_search_data, bot.total_beep, bot.crew_search_data.is_beep_recv)


    def print_all_crew_data(self, log_level, bot):
        if not self.check_log_level(log_level):
            return

        self.print_crew_data(bot.curr_pos, bot.ship, bot.crew_search_data, bot.total_beep, bot.crew_search_data.is_beep_recv)

# Modularizing our knowledge base for readability
class One_Alien_Evasion_Data:
    def __init__(self, ship):
        self.ship = ship
        self.init_alien_cells = list(self.ship.initial_alien_cells)
        self.alien_cells = list(self.ship.open_cells)
        self.alien_cells.extend([self.ship.crew_1, self.ship.crew_2, self.ship.bot])
        self.present_alien_cells = []
        self.alien_movement_cells = set()
        self.is_beep_recv = False
        self.beep_prob = 0.0
        self.beep_count = 0

        self.init_alien_cell_size = len(self.init_alien_cells)
        self.alien_cell_size = len(self.alien_cells)
        self.init_alien_calcs()

    def init_alien_calcs(self):
        # Need to change this logic (1.5 secs everytime)
        for cell_cord in self.alien_cells:
            cell = self.ship.get_cell(cell_cord)
            cell.alien_probs.alien_prob = 0 if cell.within_detection_zone else 1/self.init_alien_cell_size

    def reset_alien_calcs(self, curr_cell):
        cells_in_outer_border = self.ship.get_outer_cells(curr_cell)
        cells_within_bot_zone = self.ship.cells_within_bot_zone
        reset_cells = cells_in_outer_border + cells_within_bot_zone
        total_iter = 0
        for cell_cord in reset_cells:
            cell = self.ship.get_cell(cell_cord)
            if cell.cell_type & ~CLOSED_CELL:
                total_iter += 1
                cell.alien_probs.alien_prob = 0 if cell.within_detection_zone else 1/self.init_alien_cell_size

class Two_Alien_Evasion_Data(One_Alien_Evasion_Data):
    def __init__(self, ship):
        self.alien_cells_pair = dict()
        self.alien_cell_pair_list = list()
        super(Two_Alien_Evasion_Data, self).__init__(ship)

    def reset_alien_calcs(self):
        # reset new cell pairs to zero
        # set old cell pairs to 1/init_alien_cell_size
        len_alien_cell_pair = (self.init_alien_cell_size * (self.init_alien_cell_size - 1))/2
        for key_val in self.alien_cell_pair_list:
            cell_pair = self.alien_cells_pair[key_val]
            cell_1 = cell_pair.cell_1
            cell_2 = cell_pair.cell_2
            cell_pair.alien_probs.alien_prob = 0 if (cell_1.within_detection_zone or cell_2.within_detection_zone) else 1/len_alien_cell_pair


    def init_alien_calcs(self):
        len_alien_cell_pair = (self.init_alien_cell_size * (self.init_alien_cell_size - 1))/2
        for key_itr, key in enumerate(self.alien_cells):
            if key_itr == self.alien_cell_size - 1:
                continue

            for val_itr in range(key_itr + 1, self.alien_cell_size):
                val = self.alien_cells[val_itr]
                key_val = tuple(set((key, val)))
                cell_pair = Alien_Cell_Pair(key_val[0], key_val[1], self.ship)
                cell_1 = cell_pair.cell_1
                cell_2 = cell_pair.cell_2
                self.alien_cells_pair[key_val] = cell_pair
                cell_pair.alien_probs.alien_prob = 0 if (cell_1.within_detection_zone or cell_2.within_detection_zone) else 1/len_alien_cell_pair
                self.alien_cell_pair_list.append(key_val)


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
        self.zone_number = 0 # let the hunger games begin :p

class Cell_Pair:
    def __init__(self, cell_1, cell_2, ship):
        self.cell_1 = ship.get_cell(cell_1)
        self.cell_2 = ship.get_cell(cell_2)
        self.cells = [cell_1, cell_2]
        self.init_probs()

    def init_probs(self):
        self.crew_probs = Crew_Probs()

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
        self.all_crew_zones = {}
        for key in ship.open_cell_zones:
            self.all_crew_zones[key] = list(ship.open_cell_zones[key])

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
        self.pending_crew_to_save = 0
        self.saved_crew_cell = ()
        self.crew_cells_length = crew_cells_len * (crew_cells_len - 1)

    def update_cell_mov_vals(self, crew_probs, curr_pos, cell_cord):
        crew_probs.bot_distance = get_manhattan_distance(curr_pos, cell_cord)
        crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]

    def update_crew_pair(self, cell_pair, curr_pos):
        cell_1 = cell_pair.cell_1.cord
        cell_2 = cell_pair.cell_2.cord
        if self.pending_crew_to_save == 1:
            self.update_cell_mov_vals(cell_pair.cell_1.crew_probs, curr_pos, cell_1)
            cell_pair.cell_2.crew_probs.bot_distance = 0
            cell_pair.cell_2.crew_probs.no_beep_given_crew = 1
        elif self.pending_crew_to_save == 2:
            cell_pair.cell_1.crew_probs.bot_distance = 0
            cell_pair.cell_1.crew_probs.no_beep_given_crew = 1
            self.update_cell_mov_vals(cell_pair.cell_2.crew_probs, curr_pos, cell_2)
        else:
            self.update_cell_mov_vals(cell_pair.cell_1.crew_probs, curr_pos, cell_1)
            self.update_cell_mov_vals(cell_pair.cell_2.crew_probs, curr_pos, cell_2)

    def update_cell_pair_vals(self, cell_pair, curr_pos):
        self.update_crew_pair(cell_pair, curr_pos)
        crew_probs = cell_pair.crew_probs
        crew_probs.no_beep_given_crew = cell_pair.cell_1.crew_probs.no_beep_given_crew * cell_pair.cell_2.crew_probs.no_beep_given_crew
        crew_probs.beep_given_crew = 1 - crew_probs.no_beep_given_crew
        crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
        self.beep_prob += crew_probs.crew_and_beep
        self.no_beep_prob += crew_probs.crew_and_no_beep
        self.normalize_probs += crew_probs.crew_prob

    def print_map(self, cell):
        return

        print(cell)
        for key in self.crew_cells_pair:
            print(str(key) + "::", end="")
            length = len(self.crew_cells_pair[key])
            for itr, val in enumerate(self.crew_cells_pair[key]):
                if itr < length - 1:
                    print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_prob) + ", ", end="")
                else:
                    print(str(val.cell_2.cord) + ":" + str(val.crew_probs.crew_prob), end="")
            print()

    def update_rem_crew_probs(self, crew_pair):
        self.normalize_probs -= crew_pair.crew_probs.crew_prob
        self.crew_cells_length -= 1
        del crew_pair

    # invoke when only 1 crew member is present
    def remove_cell_probs_1(self, rem_cell, curr_pos):
        index = self.crew_cells.index(rem_cell)
        if self.pending_crew_to_save == 1: # 1 is left, so remove key here
            cells_pair_list = self.crew_cells_pair[rem_cell]
            self.update_rem_crew_probs(cells_pair_list.pop(0))
            del self.crew_cells_pair[rem_cell]
        else: # 2 is left, so remove based on value here
            cell_pair_list = self.crew_cells_pair[self.saved_crew_cell]
            for itr, cell_pair in enumerate(cell_pair_list): # a tinsy bit costly here...
                if rem_cell == cell_pair.cell_2.cord:
                    self.update_rem_crew_probs(cell_pair)
                    cell_pair_list.pop(itr)
                    break

            if not len(cell_pair_list):
                del self.crew_cells_pair[self.saved_crew_cell]

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

    def retain_success_cell_probs(self, curr_pos, pending_crew, logger):
        if curr_pos not in self.crew_cells:
            return

        index = self.crew_cells.index(curr_pos)
        if pending_crew == 1: # retain with respect to value
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
        else: # retain with respect to key
            for itr, key in enumerate(self.crew_cells):
                cells_pair_list = self.crew_cells_pair[key]
                if itr != index:
                    for val_itr in range(len(self.crew_cells) - 1):
                        self.update_rem_crew_probs(cells_pair_list.pop(0))
                    del self.crew_cells_pair[key]
                else:
                    self.update_rem_crew_probs(cells_pair_list.pop(index))

        self.pending_crew_to_save = pending_crew
        self.saved_crew_cell = curr_pos
        logger.print(LOG_DEBUG, f"Retaining following cell {curr_pos} from the list of probable crew cells {self.crew_cells}")
        self.print_map(curr_pos)

    def init_crew_calcs(self, ship, curr_pos):
        self.set_all_probs()
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
    def __init__(self, size, log_level = LOG_NONE):
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
        self.open_cell_zones = {}

        self.generate_grid()

    def get_cell(self, cord):
        return self.grid[cord[0]][cord[1]]

    def generate_grid(self):
        self.assign_start_cell()
        self.unblock_closed_cells()
        self.unblock_dead_ends()
        self.compute_adj_cells()

    def assign_zones(self):
        region_size = int(self.size/SEARCH_ZONE_SIZE) - 1
        zone_limits = {}
        zones = 0
        for i in range(0, self.size, region_size + 1):
            for j in range(0, self.size, region_size + 1):
                zone_limit = [(i,j),(i+region_size,j),(i,j+region_size),(i+region_size,j+region_size)]
                zones += 1
                zone_limits[zones] = zone_limit

        for zone in zone_limits:
            regions_cords = zone_limits[zone]
            self.open_cell_zones[zone] = []
            for i in range(regions_cords[0][0], regions_cords[3][0] + 1):
                for j in range(regions_cords[0][1], regions_cords[3][1] + 1):
                    cell = self.get_cell((i, j))
                    if cell.cell_type != CLOSED_CELL and cell.cell_type != BOT_CELL:
                        self.open_cell_zones[zone].append((i, j))
                        cell.zone_number = zone

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
        self.assign_zones()
        self.reset_detection_zone()
        self.place_aliens(2)

    def init_cell_details(self, cell_cord):
        cell = self.get_cell(cell_cord)
        cell.listen_beep.crew_1_dist = get_manhattan_distance(cell_cord, self.crew_1)
        cell.listen_beep.crew_2_dist = get_manhattan_distance(cell_cord, self.crew_2)
        cell.listen_beep.c1_beep_prob = LOOKUP_E[cell.listen_beep.crew_1_dist]
        cell.listen_beep.c2_beep_prob = LOOKUP_E[cell.listen_beep.crew_2_dist]
        cell.cord = cell_cord

    def check_aliens_count(self, no_of_aliens):
        if len(self.aliens) == no_of_aliens:
            return

        alien_cell = self.aliens.pop(1)
        alien = self.get_cell(alien_cell)
        if alien.cell_type & CREW_CELL:
            self.get_cell(alien_cell).cell_type = CREW_CELL
        else:
            self.get_cell(alien_cell).cell_type = OPEN_CELL


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

        self.aliens = list(self.initial_alien_pos)

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
                    LOG_INFO,
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
                cell = self.get_cell((row, col))
                if cell.cell_type & ~CLOSED_CELL:
                    cells_within_zone.append((row, col))

        return cells_within_zone

    def get_outer_cells(self, curr_cell = None):
        if curr_cell is None:
            curr_cell = self.bot
        outer_cells = []
        k = ALIEN_ZONE_SIZE

        # Iterate over the cells surrounding the detection zone
        for i in range(curr_cell[0] - k - 1, curr_cell[0] + k + 2):
            for j in range(curr_cell[1] - k - 1, curr_cell[1] + k + 2):
                # Check if the cell is within the grid bounds and not part of the detection zone
                if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE \
                        and (i < curr_cell[0] - k or i > curr_cell[0] + k
                            or j < curr_cell[1] - k or j > curr_cell[1] + k):
                        cell = self.get_cell((i, j))
                        if cell.cell_type & ~CLOSED_CELL:
                             outer_cells.append((i, j))

        return outer_cells

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
        self.aliens = list(self.initial_alien_pos)
        for cell in self.open_cells:
            self.get_cell(cell).cell_type = OPEN_CELL


""" Basic search algorithm, and the parent class for our bot """

class CellSearchNode:
    def __init__(self, cord, parent=None):
        self.cord = cord
        self.parent = parent
        self.actual_est = 0
        self.heuristic_est = 0
        self.total_cost = 0

    def __eq__(self, other):
        return self.cord == other.cord

    def __lt__(self, other):
        return self.total_cost < other.total_cost

class SearchAlgo:
    alien_config = ONE_ALIEN

    def __init__(self, ship, log_level):
        self.ship = ship
        self.curr_pos = ship.bot
        self.last_pos = ()
        self.logger = Logger(log_level)
        self.to_visit_list = []
        self.zone_vs_zone_prob = {}
        self.track_zones = {}
        # Working on few issues, will fix it ASAP
        self.disable_alien_calculation = True
        self.idle_threshold = IDLE_BEEP_COUNT
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
            neighbors = self.ship.get_cell(current_cell).adj_cells
            for neighbor in neighbors:
                if ((neighbor not in visited_cells) and
                    (neighbor not in unsafe_cells)):
                    bfs_queue.append((neighbor, path_traversed + [neighbor]))

        return [] #God forbid, this should never happen

    def astar_search_path(self, dest_cell, unsafe_cells):
        curr_pos = self.curr_pos

        open_list = list()
        visited_set = set()
        start_node = CellSearchNode(curr_pos)
        goal_node = CellSearchNode(dest_cell)
        heappush(open_list, start_node)

        while open_list:
            current_node = heappop(open_list)

            if current_node == goal_node:
                path = []
                while current_node:
                    path.append(current_node.cord)
                    current_node = current_node.parent
                return path[::-1]

            visited_set.add(current_node.cord)

            for neighbour in self.ship.get_cell(current_node.cord).adj_cells:
                if (neighbour in visited_set) or (neighbour in unsafe_cells):
                    continue

                new_node = CellSearchNode(neighbour, current_node)

                cell = self.ship.get_cell(new_node.cord)
                alien_prob = cell.alien_probs.alien_prob
                crew_prob = cell.crew_probs.crew_prob
                new_node.actual_est = current_node.actual_est + 1
                new_node.heuristic_est = get_manhattan_distance(new_node.cord, dest_cell)
                new_node.total_cost = new_node.actual_est - (crew_prob) + (new_node.heuristic_est * DISTANCE_UTILITY) + (2 * alien_prob) # avoid aliens!!!

                heappush(open_list, new_node)

        # No path found
        return []

    def place_aliens_handler(self):
        self.ship.check_aliens_count(self.alien_config)


""" Main parent class for all our bots, contain most of the common bot logic """

class ParentBot(SearchAlgo):
    compute_movement_config = ONE_ALIEN

    def __init__(self, ship, log_level):
        super(ParentBot, self).__init__(ship, log_level)
        self.alien_evasion_data = One_Alien_Evasion_Data(ship) # to do only in one alien cases
        # self.alien_evasion_data = One_Alien_Evasion_Data(ship) # to do only in one alien cases
        self.temp_search_data = Crew_Search_Data()
        self.crew_search_data = One_Crew_Search_DS(ship)
        self.total_crew_to_save = 2
        self.traverse_path = []
        self.unsafe_cells = []
        self.bot_escape = False
        self.pred_crew_cells = list()
        self.is_caught = False
        self.is_own_design = self.is_bot_moved = self.is_escape_strategy = self.made_move = False
        self.recalc_pred_cells = True
        self.total_beep = self.track_beep = self.pending_crew = 0
        self.logger.print_grid(self.ship.grid)
        self.path_traversed = list()
        self.path_traversed.append(self.curr_pos)
        self.all_crews = [self.ship.crew_1, self.ship.crew_2]
        self.saved_crew = ()
        self.old_path = []

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
        if not escape_path:
            return []

        escape_path.pop(0)

        return escape_path

    def find_nearest_safe_cell(self):
        curr_cell = self.ship.get_cell(self.curr_pos)
        bot_adj_cells = curr_cell.adj_cells
        safe_cells = list()

        for itr, cell_cord in enumerate(bot_adj_cells):
            cell = self.ship.get_cell(cell_cord)
            safe_cell = False
            safe_neighbours = 0
            if cell.alien_probs.alien_prob == ALIEN_NOT_PRESENT:
                if cell not in self.unsafe_cells:
                    safe_cell = True
                    for neighbour in cell.adj_cells:
                        if neighbour not in self.unsafe_cells:
                            safe_neighbours += 1

            if safe_cell:
                safe_cells.append((cell_cord, safe_neighbours))

        if len(safe_cells) > 0:
            safe_cells = sorted(safe_cells, key=lambda x: x[1], reverse=True)
            return [self.curr_pos, safe_cells[0][0]]

    '''
        If beep heard, cells within detection zone: P(B obs /A) = 1
        cells outside P(B obs /A) = 0
        If beep not heard, cells within detection zone: P(B obs /A) = 0
        cells outside P(B obs /A) = 1
        P(A/B obs) = P(B obs/ A) P(A) / P(B obs)
    '''
    def update_alien_data(self):
        if (self.alien_evasion_data.beep_count == 0):
            if self.made_move:
                self.alien_evasion_data.reset_alien_calcs(self.curr_pos)
        else:
            beep_recv = self.alien_evasion_data.is_beep_recv
            self.alien_evasion_data.beep_prob = 0
            prob_cell_list = list()
            present_alien_cells = list()

            prob_alien_in_inner_cells = ALIEN_PRESENT if beep_recv else ALIEN_NOT_PRESENT
            prob_alien_in_outer_cells = ALIEN_PRESENT if (not beep_recv) else ALIEN_NOT_PRESENT

            # Likelihood computation
            for cell_cord in self.alien_evasion_data.alien_movement_cells:
                cell = self.ship.get_cell(cell_cord)
                prob_beep_gv_alien = prob_alien_in_inner_cells if cell.within_detection_zone else prob_alien_in_outer_cells

                if (cell.cord == self.curr_pos) or (prob_beep_gv_alien == ALIEN_NOT_PRESENT):
                    cell.alien_probs.alien_prob = ALIEN_NOT_PRESENT
                    continue

                alien_prob = cell.alien_probs.alien_prob
                cell.alien_probs.alien_and_beep = prob_beep_gv_alien * alien_prob
                self.alien_evasion_data.beep_prob += cell.alien_probs.alien_and_beep
                present_alien_cells.append(cell_cord)

            is_additive_amoothing = False
            if not self.alien_evasion_data.beep_prob:
                self.alien_evasion_data.beep_prob = ADDITIVE_VALUE * len(present_alien_cells)

            # Updating the alien prob from prior knowledge
            for cell_cord in present_alien_cells:
                cell = self.ship.get_cell(cell_cord)
                if is_additive_amoothing:
                    cell.alien_probs.alien_and_beep += ADDITIVE_VALUE
                cell.alien_probs.alien_prob = cell.alien_probs.alien_and_beep/self.alien_evasion_data.beep_prob
                if cell.zone_number not in self.zone_vs_zone_prob:
                    self.zone_vs_zone_prob[cell.zone_number] = 0

                self.zone_vs_zone_prob[cell.zone_number] += ALIEN_UTILITY*cell.alien_probs.alien_prob
                dist_cell = get_manhattan_distance(self.curr_pos, cell_cord)
                prob_cell_list.append((cell.alien_probs.alien_prob, cell_cord, dist_cell))

            # Sorting by probability, used to track movements under limitation etc
            cells_by_distance = sorted(prob_cell_list, key=lambda x: x[2], reverse=True)
            self.alien_evasion_data.present_alien_cells = [cell[1] for cell in cells_by_distance]
            if self.alien_evasion_data.beep_count > 0:
                prob_cell_list = sorted(prob_cell_list, key=lambda x: x[0], reverse=True)
                self.unsafe_cells = [cell[1] for cell in prob_cell_list][:ALIEN_ZONE_SIZE+1]
                # if not beep_recv:
                #     self.unsafe_cells = self.unsafe_cells[:TOTAL_UNSAFE_CELLS]

                unsafe_neighbours = list()
                for cell_cord in self.unsafe_cells:
                    cell = self.ship.get_cell(cell_cord)
                    unsafe_neighbours.append(cell.adj_cells)
                self.unsafe_cells.extend(unsafe_neighbours)


    def compute_likely_alien_movements(self):
        beep_recv = self.alien_evasion_data.is_beep_recv
        beep_count = self.alien_evasion_data.beep_count

        if (beep_count == 1) and beep_recv:
            self.alien_evasion_data.present_alien_cells = self.ship.get_outer_cells(self.curr_pos) + self.ship.cells_within_bot_zone

        self.alien_evasion_data.alien_movement_cells = set()
        prob_cell_mapping = dict()

        present_alien_cells = self.alien_evasion_data.present_alien_cells


        for cell_cord in present_alien_cells:
            self.logger.print(
                LOG_DEBUG, f"Iterating for ::{cell_cord}"
            )
            cell = self.ship.get_cell(cell_cord)
            possible_moves = cell.adj_cells
            total_moves = len(possible_moves)

            if (total_moves == 0):
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
                self.alien_evasion_data.alien_movement_cells.add(alien_move)
                adj_cell.alien_probs.alien_prob += prob_cell_mapping[cell.cord]/total_moves

        prob_cell_mapping.clear()


    """
        Ideally it is better to move the bot in the direction of the highest prob
        To do this, pred_crew_cells should be sorted based on probabilty
        Remember, we are not taking into account where the alien will be here!!
    """
    def move_bot(self):
        if not self.traverse_path:
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

        # self.logger.print(LOG_NONE, f"Bot {self.last_pos} has moved to {self.curr_pos} trying to find {self.total_crew_to_save} crew pending")
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
                self.saved_crew = self.curr_pos
                break

        if len(self.all_crews) == 0:
            return True

        return False

    def remove_cell(self, rem_cell):
        return

    def norm_probs(self):
        return

    def remove_nearby_cells(self):
        crew_search_data = self.crew_search_data
        neighbors = self.ship.get_cell(self.curr_pos).adj_cells
        for neighbor in neighbors:
            self.remove_cell(neighbor)

    def update_crew_search_data(self):
        crew_search_data = self.crew_search_data
        if not crew_search_data.is_beep_recv: # no beep heard
            self.remove_nearby_cells()

        if self.is_bot_moved or (crew_search_data.normalize_probs < 1): # normalize our probs
            self.norm_probs()
            self.is_bot_moved = False

        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def rem_cell_for_crew_search(self, rem_cell):
        zone_number = rem_cell.zone_number
        if zone_number in self.crew_search_data.all_crew_zones:
            self.crew_search_data.all_crew_zones[zone_number].remove(rem_cell.cord)
            if len(self.crew_search_data.all_crew_zones[zone_number]) == 0:
                del self.crew_search_data.all_crew_zones[zone_number]

        rem_cell.crew_probs.crew_prob = 0 # can't use this as our if condition because the crew_probs tend to 0 sometimes
        self.crew_search_data.crew_cells.remove(rem_cell.cord)

    def update_cell_crew_probs(self, crew_probs):
        crew_probs.crew_given_beep = crew_probs.crew_and_beep/self.crew_search_data.beep_prob
        if self.crew_search_data.no_beep_prob: # it is possible to never not hear beeps depending on how far we have searched. can norm if required.
            crew_probs.crew_given_no_beep = crew_probs.crew_and_no_beep/self.crew_search_data.no_beep_prob
        crew_probs.crew_prob = crew_probs.crew_given_beep if self.crew_search_data.is_beep_recv else crew_probs.crew_given_no_beep
        crew_probs.crew_and_beep = crew_probs.beep_given_crew*crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew*crew_probs.crew_prob
        self.temp_search_data.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

    def get_best_zone(self):
        zone_as_list = []
        self.track_zones.clear()
        curr_zone = self.ship.get_cell(self.curr_pos).zone_number
        curr_zone_pos = (curr_zone%SEARCH_ZONE_SIZE, floor(curr_zone/SEARCH_ZONE_SIZE))
        for key in self.zone_vs_zone_prob:
            # distance = abs(curr_zone_pos[0] - key%SEARCH_ZONE_SIZE) + abs(curr_zone_pos[1] - floor(key/SEARCH_ZONE_SIZE))
            # self.zone_vs_zone_prob[key] -= DISTANCE_UTILITY*distance #don't want distance to play a major factor...

            if key in self.crew_search_data.all_crew_zones:
                zone_as_list.append((key, self.zone_vs_zone_prob[key]))

        zone_as_list = sorted(zone_as_list, key=lambda data:-data[1])
        if not zone_as_list:
            return sorted(self.crew_search_data.crew_cells, key= lambda cell:(-self.ship.get_cell(cell).crew_probs.crew_prob, self.ship.get_cell(cell).crew_probs.bot_distance))[:3]

        # point to check, is it worth visiting a zone, if the zone value changes???
        max_size = 8 if self.total_crew_to_save == 2 else 6
        zone_1 = sorted(self.crew_search_data.all_crew_zones[zone_as_list[0][0]], key=lambda cell:(-self.ship.get_cell(cell).crew_probs.crew_prob, self.ship.get_cell(cell).crew_probs.bot_distance))[:max_size]
        self.track_zones[zone_as_list[0][0]] = (zone_1, self.zone_vs_zone_prob[zone_as_list[0][0]])
        return zone_1

    def get_most_prob_cell(self):
        # zone_number = self.ship.get_cell(self.curr_pos).zone_number
        # next_best_crew_cells = []
        # if zone_number in self.crew_search_data.all_crew_zones:
        #     next_best_crew_cells = sorted(self.crew_search_data.all_crew_zones[zone_number], key=lambda cell:(-self.ship.get_cell(cell).crew_probs.crew_prob, self.ship.get_cell(cell).crew_probs.bot_distance))[:3]

        # most_prob_cells = sorted(self.crew_search_data.crew_cells, key= lambda cell:(-self.ship.get_cell(cell).crew_probs.crew_prob, self.ship.get_cell(cell).crew_probs.bot_distance))[:3]
        # next_best_crew_cells.extend(most_prob_cells)

        next_best_crew_cells = sorted(self.crew_search_data.crew_cells, key= lambda cell:(self.ship.get_cell(cell).crew_probs.bot_distance, -self.ship.get_cell(cell).crew_probs.crew_prob))[:4]
        more_prob_cells = sorted(self.crew_search_data.crew_cells, key= lambda cell:(-self.ship.get_cell(cell).crew_probs.crew_prob, self.ship.get_cell(cell).crew_probs.bot_distance))[:3]
        next_best_crew_cells.extend(more_prob_cells)
        return next_best_crew_cells

    def is_continue_traversing(self):
        if not self.is_own_design or len(self.to_visit_list) == 0 or len(self.track_zones) == 0:
            return False

        dest = self.to_visit_list[0]
        zone_number = self.ship.get_cell(dest).zone_number
        if zone_number not in self.track_zones:
            return False

        if not zone_number in self.zone_vs_zone_prob:
            for element in self.track_zones[zone_number][0]:
                if element in self.to_visit_list:
                    self.to_visit_list.remove(element)

            del self.track_zones[zone_number]
            return False

        if self.zone_vs_zone_prob[zone_number] > self.track_zones[zone_number][1]:
            total_elements = len(self.track_zones[zone_number][0])
            common_elements = []
            for x, y in zip(self.to_visit_list, self.track_zones[zone_number][0]):
                if x == y:
                    common_elements.append(x)

            if len(common_elements) < total_elements * .75:
                for element in common_elements:
                    self.to_visit_list.remove(element)

                del self.track_zones[zone_number]
                return False

        return True

    def calculate_best_path(self):
        if self.is_own_design and self.traverse_path:
            self.traverse_path = self.astar_search_path(self.traverse_path[-1], self.unsafe_cells)

            if self.traverse_path:
                self.traverse_path.pop(0)
                return True

        if self.traverse_path and (self.traverse_path in self.unsafe_cells): # Escape path
            self.traverse_path = self.search_path(self.traverse_path[-1], None, self.unsafe_cells)

            if self.traverse_path:
                self.traverse_path.pop(0)
                return True
            else:
                if not self.old_path:
                    self.old_path = list(self.traverse_path)

                self.traverse_path = self.find_escape_path()
                if self.traverse_path:
                    return True
                return False

        if self.traverse_path:
            return True
        elif self.old_path:
            self.traverse_path = self.old_path
            self.old_path.clear()
            if self.traverse_path:
                return True

        self.is_continue_traversing()

        if len(self.to_visit_list) == 0:
            if self.is_own_design:
                dummy, self.to_visit_list = self.shortest_route(self.get_best_zone())
            else:
                dummy, self.to_visit_list = self.shortest_route(self.get_most_prob_cell())
            index = self.to_visit_list.index(self.curr_pos)
            self.to_visit_list.pop(index)
            if index > 0:
                self.to_visit_list.reverse()

        prob_crew_cell = self.to_visit_list.pop(0)
        if self.is_own_design:
            self.traverse_path = self.astar_search_path(prob_crew_cell, self.unsafe_cells)
        else:
            if len(self.unsafe_cells):
                self.traverse_path = self.search_path(prob_crew_cell, None, self.unsafe_cells)

        if self.traverse_path:
            self.traverse_path.pop(0)
            return True

        if (not self.alien_evasion_data.is_beep_recv):
            self.traverse_path = self.search_path(prob_crew_cell)

        if not self.traverse_path:
            self.logger.print(LOG_DEBUG, f"Unable to find a path....")
            return False

        self.traverse_path.pop(0)
        self.logger.print(LOG_DEBUG, f"New path to cell {prob_crew_cell} was found, {self.traverse_path}")
        return True

    def get_saved(self):
        saved = self.total_crew_to_save - len(self.all_crews)
        if self.total_crew_to_save == 1 and len(self.all_crews) == 1:
            saved = 1
        elif self.total_crew_to_save == 1 and len(self.all_crews) == 2:
            saved = 0

        return saved

    def start_rescue(self): # working, finalllly ;-;
        total_moves = idle_steps = total_iter = 0
        keep_moving = False
        init_distance = self.rescue_info()
        self.calc_initial_search_data()
        self.logger.print_all_crew_data(LOG_DEBUG, self)

        while (True): # Keep trying till you find the crew
            if total_iter >= 1000:
                return init_distance, total_iter, total_moves, BOT_STUCK, self.get_saved()

            total_iter += 1
            idle_steps += 1
            self.handle_alien_beep()
            self.handle_crew_beep()

            if self.alien_evasion_data.beep_count > 0:
                self.compute_likely_alien_movements()

            self.zone_vs_zone_prob.clear()
            self.update_alien_data()
            self.update_crew_search_data()

            # if idle_steps >= self.idle_threshold or self.alien_evasion_data.beep_count > 0:
            #     idle_steps = 0
            #     keep_moving = self.calculate_best_path()

            if self.calculate_best_path():
                if self.move_bot():
                    if self.is_rescued():
                        return init_distance, total_iter, total_moves, BOT_SUCCESS, self.get_saved()

                    elif self.is_caught:
                        return init_distance, total_iter, total_moves, BOT_FAILED, self.get_saved()

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
                return init_distance, total_iter, total_moves, BOT_FAILED, self.get_saved()

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

    def remove_cell(self, rem_cell):
        crew_search_data = self.crew_search_data
        if rem_cell not in crew_search_data.crew_cells:
            return

        cell = self.ship.get_cell(rem_cell)
        crew_probs = cell.crew_probs
        crew_search_data.normalize_probs -= crew_probs.crew_prob
        self.rem_cell_for_crew_search(cell)
        self.logger.print(LOG_DEBUG, f"Removed {rem_cell} cell from possible crew cells {crew_search_data.crew_cells}")

    def norm_probs(self):
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()
        alpha_num = 0.0
        if crew_search_data.normalize_probs <= 0:
            self.logger.print_all_crew_data(LOG_DEBUG, self)
            self.logger.print(LOG_DEBUG, "Using additive smoothing...")
            alpha_num = ADDITIVE_VALUE
            crew_search_data.normalize_probs = alpha_num * len(crew_search_data.crew_cells)
        elif crew_search_data.normalize_probs > 1.5:
            self.logger.print_all_crew_data(LOG_NONE, self)
            print(self.__class__.__name__, "THERE WAS A MAJOR NEWS!!!!")
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

    def update_crew_search_data(self):
        super(Bot_1, self).update_crew_search_data()
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()

        for cell_cord in crew_search_data.crew_cells: # update probs for this round
            cell = self.ship.get_cell(cell_cord)
            crew_probs = cell.crew_probs
            self.update_cell_crew_probs(crew_probs)
            if cell.zone_number not in self.zone_vs_zone_prob:
                self.zone_vs_zone_prob[cell.zone_number] = 0

            if cell.zone_number in self.crew_search_data.all_crew_zones:
                self.zone_vs_zone_prob[cell.zone_number] += CREW_UTILITY*crew_probs.crew_prob

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def is_rescued(self): # since this will only be used by 1, 2 and 3 (reason for if condition), this has been overriden
        if self.total_crew_count == 1:
            if self.curr_pos == self.ship.crew_1:
                self.logger.print(LOG_INFO, f"Bot has saved crew member at cell {self.ship.crew_1}")
                self.all_crews.remove(self.curr_pos)
                return True
            return False
        else:
            return super(Bot_1, self).is_rescued()

class Bot_2(Bot_1):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_2, self).__init__(ship, log_level)
        self.is_own_design = True
        self.idle_threshold = 0

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

    def remove_cell(self, rem_cell):
        crew_search_data = self.crew_search_data
        if rem_cell not in crew_search_data.crew_cells:
            return

        if self.pending_crew: # when only one crew is pending, we remove crew differently
            crew_search_data.remove_cell_probs_1(rem_cell, self.curr_pos)
        else: # when only both crews are present, gotta remove them all
            crew_search_data.remove_cell_probs_2(rem_cell, self.curr_pos)

        cell = self.ship.get_cell(rem_cell)
        self.rem_cell_for_crew_search(cell)

    def norm_probs(self):
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()
        alpha_num = 0.0
        if crew_search_data.normalize_probs <= 0:
            crew_search_data.print_map(self.curr_pos)
            self.logger.print(LOG_DEBUG, "Using additive smoothing...")
            alpha_num = ADDITIVE_VALUE
            crew_search_data.normalize_probs = alpha_num * crew_search_data.crew_cells_length
        elif crew_search_data.normalize_probs > 1.5:
            self.logger.print_all_pair_data(LOG_NONE, self)
            print(self.__class__.__name__, self.pending_crew, self.ship.crew_1, self.ship.crew_2, self.all_crews, "THERE WAS A MAJOR NEWS!!!!")
            exit(0)

        for key in self.crew_search_data.crew_cells_pair:
            cell_pair_list = self.crew_search_data.crew_cells_pair[key]
            for cell_pair in cell_pair_list:
                crew_probs = cell_pair.crew_probs
                if alpha_num:
                    crew_probs.crew_prob = crew_probs.crew_prob + alpha_num

                crew_probs.crew_prob = crew_probs.crew_prob / crew_search_data.normalize_probs
                if (self.is_bot_moved):
                    crew_search_data.update_crew_pair(cell_pair, self.curr_pos)
                    crew_probs.no_beep_given_crew = cell_pair.cell_1.crew_probs.no_beep_given_crew * cell_pair.cell_2.crew_probs.no_beep_given_crew
                    crew_probs.beep_given_crew = 1 - crew_probs.no_beep_given_crew

                crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
                crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
                temp_search_data.update_all_probs(crew_probs.crew_and_beep, crew_probs.crew_and_no_beep, crew_probs.crew_prob)

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_pair_data(LOG_DEBUG, self)

    def update_crew_search_data(self):
        super(Bot_4, self).update_crew_search_data()
        crew_search_data = self.crew_search_data
        temp_search_data = self.temp_search_data
        temp_search_data.set_all_probs()
        is_visited = {}

        # crew_cells_len can be one after saving a crew, if not,
        # the number of time a prob is calc for a cell will be n times, n being the size of the grid
        crew_cells_len = 1 if self.pending_crew else len(self.crew_search_data.crew_cells)
        for key in crew_search_data.crew_cells_pair: # update probs for this round
            for cell_pairs in crew_search_data.crew_cells_pair[key]:
                crew_probs = cell_pairs.crew_probs
                self.update_cell_crew_probs(crew_probs)
                cell_1 = cell_pairs.cell_1
                cell_2 = cell_pairs.cell_2

                if cell_1.cord not in is_visited:
                    is_visited[cell_1.cord] = 1
                    cell_1.crew_probs.crew_prob = 0.0

                if cell_2.cord not in is_visited:
                    is_visited[cell_2.cord] = 1
                    cell_2.crew_probs.crew_prob = 0.0

                if cell_1.cord != self.saved_crew:
                    if cell_1.zone_number not in self.zone_vs_zone_prob:
                        self.zone_vs_zone_prob[cell_1.zone_number] = 0.0

                    cell_1.crew_probs.crew_prob += crew_probs.crew_prob

                    if cell_1.zone_number in self.crew_search_data.all_crew_zones: #double checking to be safe...
                        self.zone_vs_zone_prob[cell_1.zone_number] += CREW_UTILITY*crew_probs.crew_prob/crew_cells_len

                if cell_2.cord != self.saved_crew:
                    if cell_2.zone_number not in self.zone_vs_zone_prob:
                        self.zone_vs_zone_prob[cell_2.zone_number] = 0.0

                    cell_2.crew_probs.crew_prob += crew_probs.crew_prob

                    if cell_2.zone_number in self.crew_search_data.all_crew_zones:
                        self.zone_vs_zone_prob[cell_2.zone_number] += CREW_UTILITY*crew_probs.crew_prob/crew_cells_len

        crew_search_data.set_all_probs(temp_search_data.beep_prob, temp_search_data.no_beep_prob, temp_search_data.normalize_probs)
        self.logger.print_all_crew_data(LOG_DEBUG, self)

    def is_rescued(self):
        old_val = self.pending_crew
        ret_val = super(Bot_4, self).is_rescued()
        if old_val != self.pending_crew and len(self.all_crews):
            self.crew_search_data.retain_success_cell_probs(self.curr_pos, self.pending_crew, self.logger)
            self.rem_cell_for_crew_search(self.ship.get_cell(self.curr_pos))
        return ret_val

class Bot_5(Bot_4):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_5, self).__init__(ship, log_level)
        self.is_own_design = True
        self.idle_threshold = 0

class Bot_6(Bot_3):
    alien_config = TWO_ALIENS

    def __init__(self, ship, log_level=LOG_NONE):
        super(Bot_6, self).__init__(ship, log_level)

class Bot_7(Bot_4):
    alien_config = TWO_ALIENS

    def __init__(self, ship, log_level=LOG_NONE):
        super(Bot_7, self).__init__(ship, log_level)
        self.alien_evasion_data = Two_Alien_Evasion_Data(self.ship)

    def get_initial_compute_cell_pairs(self):
        return self.alien_evasion_data.alien_cell_pair_list

    def compute_likely_alien_movements(self):
        start = time()
        beep_recv = self.alien_evasion_data.is_beep_recv
        beep_count = self.alien_evasion_data.beep_count

        if (beep_count == 1) and beep_recv:
            self.present_alien_cells = self.get_initial_compute_cell_pairs()

        self.alien_evasion_data.alien_movement_cells = set()

        present_alien_cells = self.alien_evasion_data.present_alien_cells

        prob_cell_pair_mapping = dict()

        for key_val_cells in present_alien_cells:
            possible_moves = dict()
            cell_pair = self.alien_evasion_data.alien_cells_pair[key_val_cells]
            no_moves = 0
            self.logger.print(
                LOG_DEBUG, f"Iterating for ::{cell.cord}"
            )
            cell_1 = cell_pair.cell_1
            cell_2 = cell_pair.cell_2

            total_moves = 1
            for cell in [cell_1, cell_2]:
                if len(cell.adj_cells) == 0:
                    no_moves += 1
                    continue
                possible_moves[cell.cord] = cell.adj_cells
                total_moves *= len(possible_moves[cell.cord])

            if ((no_moves == 2)):
                continue

            if key_val_cells not in prob_cell_pair_mapping.keys():
                prob_cell_pair_mapping[key_val_cells] = cell_pair.alien_probs.alien_prob
                cell_pair.alien_probs.alien_prob = 0

            self.logger.print(
                LOG_DEBUG, f"prob_cell_mapping::{prob_cell_pair_mapping}"
            )

            self.logger.print(
                LOG_DEBUG, f"Neighbours for the current cell pair::{possible_moves}"
            )

            # Cell pair movement logic
            for alien_moves_1 in possible_moves[cell_1.cord]:
                for alien_moves_2 in possible_moves[cell_2.cord]:
                    if alien_moves_1 == alien_moves_2:
                        continue
                    adj_key_val_cell = tuple(set(alien_moves_1, alien_moves_2))
                    adj_cell_pair = self.alien_evasion_data.alien_cells_pair[adj_key_val_cell]
                    if adj_key_val_cell not in prob_cell_pair_mapping.keys():
                        prob_cell_pair_mapping[adj_key_val_cell] = adj_cell_pair.alien_probs.alien_prob
                        adj_cell_pair.alien_probs.alien_prob = 0
                    self.alien_evasion_data.alien_movement_cells.add(adj_key_val_cell)
                    adj_cell_pair.alien_probs.alien_prob += prob_cell_pair_mapping[cell.cord]/total_moves

            possible_moves.clear()

        prob_cell_pair_mapping.clear()
        end = time()

    def update_alien_data(self):
        if (self.alien_evasion_data.beep_count == 0):
            if self.made_move:
                self.alien_evasion_data.reset_alien_calcs()
        else:
            start = time()
            beep_recv = self.alien_evasion_data.is_beep_recv
            self.alien_evasion_data.beep_prob = 0
            prob_cell_list = list()
            present_alien_cells = list()
            visited_cells = set()

            prob_alien_in_inner_cells = ALIEN_PRESENT if beep_recv else ALIEN_NOT_PRESENT
            prob_alien_in_outer_cells = ALIEN_PRESENT if (not beep_recv) else ALIEN_NOT_PRESENT

            # Likelihood computation
            for cell_pair_key in self.alien_evasion_data.alien_movement_cells:
                cell_pair = self.alien_evasion_data.alien_cells_pair[cell_pair_key]
                cell_1 = cell_pair.cell_1
                cell_2 = cell_pair.cell_2
                cell_1.alien_probs.alien_prob = 0
                cell_2.alien_probs.alien_prob = 0


                prob_beep_gv_alien = prob_alien_in_inner_cells if (cell_1.within_detection_zone or cell_2.within_detection_zone) else prob_alien_in_outer_cells
                if (self.curr_pos in cell_pair.cells) or (prob_beep_gv_alien == ALIEN_NOT_PRESENT):
                    cell_pair.alien_probs.alien_prob = ALIEN_NOT_PRESENT
                    continue

                alien_prob = cell_pair.alien_probs.alien_prob
                cell_pair.alien_probs.alien_and_beep = prob_beep_gv_alien * alien_prob
                self.alien_evasion_data.beep_prob += cell_pair.alien_probs.alien_and_beep
                present_alien_cells.append(cell_pair_key)

            # Updating the alien prob from prior knowledge
            for cell_pair_key in present_alien_cells:
                cell_pair = self.alien_evasion_data.alien_cells_pair[cell_pair_key]
                cell_pair.alien_probs.alien_prob = cell_pair.alien_probs.alien_and_beep/self.alien_evasion_data.beep_prob
                for cell_cord in cell_pair.cells:
                    # dist_cell = get_manhattan_distance(self.curr_pos, cell_cord)
                    prob = cell_pair.alien_probs.alien_prob
                    cell = self.ship.get_cell(cell_cord)
                    cell.alien_probs.alien_prob += prob
                    visited_cells.add(cell)

            total_len = len(visited_cells)
            for cell_cord in visited_cells:
                dist_cell = get_manhattan_distance(self.curr_pos, cell_cord)
                cell = self.ship.get_cell(cell_cord)
                cell.alien_probs.alien_prob /= total_len
                prob_cell_list.append((cell.alien_probs.alien_prob, cell_cord, dist_cell))

                if cell.zone_number not in self.zone_vs_zone_prob:
                    self.zone_vs_zone_prob[cell.zone_number] = 0

                self.zone_vs_zone_prob[cell.zone_number] += ALIEN_UTILITY*cell.alien_probs.alien_prob/total_len

            # Sorting by probability, used to track movements under limitation etc
            # cells_by_distance = sorted(prob_cell_list, key=lambda x: x[2], reverse=True)
            self.alien_evasion_data.present_alien_cells = present_alien_cells
            if self.alien_evasion_data.beep_count > 0:
                prob_cell_list = sorted(prob_cell_list, key=lambda x: x[0], reverse=True)
                self.unsafe_cells = [cell[1] for cell in prob_cell_list][:ALIEN_ZONE_SIZE+1]
                # if not beep_recv:
                #     self.unsafe_cells = self.unsafe_cells

                unsafe_neighbours = list()
                for cell_cord in self.unsafe_cells:
                    cell = self.ship.get_cell(cell_cord)
                    unsafe_neighbours.append(cell.adj_cells)
                self.unsafe_cells.extend(unsafe_neighbours)

            end = time()

class Bot_8(Bot_7):
    alien_config = TWO_ALIENS

    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_8, self).__init__(ship, log_level)
        self.is_own_design = True
        self.idle_threshold = 0



"""Simulation & Testing logic begins"""

BOT_NAMES = {
    0 : "bot_1",
    1 : "bot_2",
    2 : "bot_3",
    3 : "bot_4",
    4 : "bot_5",
    5 : "bot_6",
    6 : "bot_7",
    7 : "bot_8"
}

# Responsible for updating the alpha for each worker pool
def update_lookup(data, is_k):
    global LOOKUP_E, LOOKUP_NOT_E, ALPHA, ALIEN_ZONE_SIZE
    if is_k:
        ALIEN_ZONE_SIZE = data
    else:
        ALPHA = data

    LOOKUP_E = [(pow(exp, (-1*ALPHA*(i - 1)))) for i in range(GRID_SIZE*2 + 1)]
    LOOKUP_NOT_E = [(1-LOOKUP_E[i]) for i in range(GRID_SIZE*2 + 1)]

def bot_factory(itr, ship, log_level = LOG_NONE):
    if (itr == 0):
        return Bot_1(ship, log_level)
    elif (itr == 1):
        return Bot_2(ship, log_level)
    elif (itr == 2):
        return Bot_3(ship, log_level)
    elif (itr == 3):
        return Bot_4(ship, log_level)
    elif (itr == 4):
        return Bot_5(ship, log_level)
    elif (itr == 5):
        return Bot_6(ship, log_level)
    elif (itr == 6):
        return Bot_7(ship, log_level)
    elif (itr == 7):
        return Bot_8(ship, log_level)
    return ParentBot(ship, log_level)

# Test function
def run_test(log_level = LOG_INFO):
    update_lookup(ALPHA, False)
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
        self.success_steps = 0
        self.failure = 0
        self.failure_steps = 0
        self.stuck = 0
        self.stuck_steps = 0
        self.time_taken = 0.0
        self.crews_saved = 0
        pass

# Runs n number of iteration for each bot for given alpha value
def run_sim(args):
    iterations_range = args[0]
    data_range = args[1]
    data_dict = dict()
    itr_data = []
    is_k = True
    if "alpha" in data_range:
        itr_data = data_range["alpha"]
        is_k = False
    else:
        itr_data = data_range["k"]

    # varying_data = "k" if is_k else "alpha"
    for data in itr_data:
        update_lookup(data, is_k)
        # print(f"Verifying update (alpha vs k) for variable {varying_data}::{ALPHA}::{ALIEN_ZONE_SIZE}")
        temp_data_set = [FINAL_OUT() for j in range(TOTAL_BOTS)]
        for itr in range(iterations_range):
            # print(itr+1, end = '\r') # MANNNYYY LINES PRINTED ON ILAB ;-;
            ship = Ship(GRID_SIZE)
            ship.place_players()
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
                    temp_data_set[bot_no].success_steps += ret_vals[1]
                elif ret_vals[3] == BOT_FAILED:
                    temp_data_set[bot_no].failure += 1
                    temp_data_set[bot_no].failure_steps += ret_vals[1]
                else:
                    temp_data_set[bot_no].stuck += 1
                    temp_data_set[bot_no].stuck_steps += ret_vals[1]
                temp_data_set[bot_no].crews_saved += ret_vals[4]
                temp_data_set[bot_no].time_taken += (end-begin)
                ship.reset_grid()
                del bot
            del ship
        data_dict[data] = temp_data_set
    return data_dict

# Creates "n" process, and runs multiple simulation for same value of alpha simulataenously
def run_multi_sim(data_range, is_print = False):
    begin = time()
    result_dict = dict()
    data_set = [FINAL_OUT() for j in range(TOTAL_BOTS)]
    processes = []
    print(f"Iterations begin...")
    core_count = cpu_count()
    total_iters = round(TOTAL_ITERATIONS/core_count)
    actual_iters = total_iters * core_count
    total_data = []
    for itr in range(core_count):
        total_data.append((total_iters, data_range))

    with Pool(processes=core_count) as p:
        result = p.map(run_sim, total_data)
        for temp_alpha_dict in result:
            for key, value in temp_alpha_dict.items():
                if key not in result_dict:
                    result_dict[key] = value
                else:
                    for i, val_range in enumerate(value):
                        result_dict[key][i].distance += value[i].distance
                        result_dict[key][i].total_iter += value[i].total_iter
                        result_dict[key][i].total_moves += value[i].total_moves
                        result_dict[key][i].success += value[i].success
                        result_dict[key][i].success_steps += value[i].success_steps
                        result_dict[key][i].failure += value[i].failure
                        result_dict[key][i].failure_steps += value[i].failure_steps
                        result_dict[key][i].stuck += value[i].stuck
                        result_dict[key][i].stuck_steps += value[i].stuck_steps
                        result_dict[key][i].crews_saved += value[i].crews_saved
                        result_dict[key][i].time_taken += value[i].time_taken

    for key, resc_val in result_dict.items():
        for itr, value in enumerate(resc_val):
            value.distance /= actual_iters
            value.total_iter /= actual_iters
            value.total_moves /= actual_iters
            value.idle_moves = value.total_iter - value.total_moves
            if value.success:
                value.success_steps /= value.success
            value.success /= actual_iters
            if value.failure:
                value.failure_steps /= value.failure
            value.failure /= actual_iters
            if value.stuck:
                value.stuck_steps /= value.stuck
            value.stuck /= actual_iters
            value.crews_saved /= actual_iters
            value.time_taken /= actual_iters
    end = time()

    is_const_alpha = True
    if "alpha" in data_range:
        is_const_alpha = False

    if (is_print):
        for key, resc_val in result_dict.items():
            if is_const_alpha:
                print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for k {key}, and alpha {ALPHA}")
            else:
                print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {key}, and k {ALIEN_ZONE_SIZE}")
            print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" % ("Bot", "Success Rate", "Failure Rate", "Stuck", "Distance", "Crews Saved", "Success steps", "Failure steps", "Stuck steps", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
            for itr, value in enumerate(resc_val):
                print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" % (BOT_NAMES[itr], value.success, value.failure, value.stuck, value.distance, value.crews_saved, value.success_steps, value.failure_steps, value.stuck_steps, value.total_iter, value.idle_moves, value.total_moves, value.time_taken))
    else:
        if is_const_alpha:
            print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for k {result_dict.keys()}, and alpha {ALPHA}")
        else:
            print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {result_dict.keys()}, and k {ALIEN_ZONE_SIZE}")

    del processes
    return result_dict

# Runs multiple simulations for multiple values of alpha concurrently
def compare_multiple_alpha():
    global ALPHA
    ALPHA = 0.05
    alpha_range = [round(ALPHA + (ALPHA_STEP_INCREASE * i), 2) for i in range(MAX_ALPHA_ITERATIONS)]
    alpha_dict = run_multi_sim({"alpha" : alpha_range})
    print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" % ("Bot", "Success Rate", "Failure Rate", "Stuck", "Distance", "Crews Saved", "Success steps", "Failure steps", "stuck Steps", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
    for alpha, resc_val in alpha_dict.items():
        print(f"{'*'*82}{alpha}{'*'*82}")
        for itr, value in enumerate(resc_val):
            print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" %  (BOT_NAMES[itr], value.success, value.failure, value.stuck, value.distance, value.crews_saved, value.success_steps, value.failure_steps, value.stuck_steps, value.total_iter, value.idle_moves, value.total_moves, value.time_taken))

def compare_multiple_k():
    global ALIEN_ZONE_SIZE
    ALIEN_ZONE_SIZE = 2
    k_range = [round(ALIEN_ZONE_INCREASE + (ALIEN_ZONE_INCREASE * i), 2) for i in range(MAX_K_ITERATIONS)]
    k_dict = run_multi_sim({"k" : k_range})
    print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" % ("Bot", "Success Rate", "Failure Rate", "Stuck", "Distance", "Crews Saved", "Success steps", "Failure steps", "stuck Steps", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
    for k, resc_val in k_dict.items():
        print(f"{'*'*82}{k}{'*'*82}")
        for itr, value in enumerate(resc_val):
            print ("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s %20s" %  (BOT_NAMES[itr], value.success, value.failure, value.stuck, value.distance, value.crews_saved, value.success_steps, value.failure_steps, value.stuck_steps, value.total_iter, value.idle_moves, value.total_moves, value.time_taken))

# MAJOR ISSUES WITH ALL BOTS!!
if __name__ == '__main__':
    run_test()
    # run_multi_sim({"alpha" : [ALPHA]}, True)
    # run_multi_sim({"k" : [ALIEN_ZONE_SIZE]}, True)
    # compare_multiple_alpha()
    # compare_multiple_k()
