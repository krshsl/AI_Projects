from random import randint, uniform, choice
from matplotlib import pyplot
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
BOT_CAUGHT_CELL = 32
ALIEN_MOVEMENT_CELLS = CREW_CELL | OPEN_CELL # BOT_CELL
GRID_SIZE = 35

X_COORDINATE_SHIFT = [1, 0, 0, -1]
Y_COORDINATE_SHIFT = [0, 1, -1, 0]

ALIEN_ZONE_SIZE = 3 # k - k >= 1, need to determine the large value
ALPHA = 2 # avoid alpha > 11 for 35x35
IDLE_BEEP_COUNT = 10

TOTAL_ITERATIONS = 1000
MAX_ALPHA_ITERATIONS = 10
ALPHA_STEP_INCREASE = 0.2
TOTAL_BOTS = 1

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

    def print_cell_data(self, log_level, cell, curr_pos):
        if self.check_log_level(log_level):
            print(currentframe().f_back.f_code.co_name, "::", currentframe().f_back.f_lineno, "::", sep="", end="")
            print(f"curr_pos::{curr_pos}, cell_cord::{cell.cord}, cell_distance::{cell.crew_probs.bot_distance}, crew_prob::{cell.crew_probs.crew_prob}, beep_given_crew::{cell.crew_probs.beep_given_crew}, no_beep_given_crew::{cell.crew_probs.no_beep_given_crew}, crew_and_beep::{cell.crew_probs.crew_and_beep}, crew_and_no_beep::{cell.crew_probs.crew_and_no_beep}")

    def print_grid(self, grid):
        if not self.check_log_level(LOG_DEBUG_GRID):
            return

        print("****************")
        for i, cells in enumerate(grid):
            for j, cell in enumerate(cells):
                print(f"{i}{j}::{cell.cell_type}", end = " ")
            print("")
        print("****************")

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


# Modularizing our knowledge base for readbility
class Alien_Evasion_Data:
    def __init__(self, ship):
        self.present_alien_cells = list(ship.initial_alien_cells)
        self.all_alien_cells = list(ship.open_cells)
        self.all_alien_cells.extend([ship.crew_1, ship.bot])

class Crew_Search_Data:
    def __init__(self):
        self.beep_prob = 0 # p(b) -> normalized for hearing beeps
        self.no_beep_prob = 0 # p(¬b) -> normalized for hearing no beeps
        self.beep_count = 0
        self.is_beep_recv = False

class One_Crew_Search(Crew_Search_Data):
    def __init__(self, ship):
        super(One_Crew_Search, self).__init__()
        self.crew_cells = list(ship.open_cells) # list of all possible crew cells
        self.crew_cells.append(ship.crew_1)
        self.crew_cells.append(ship.crew_2)

class Two_Crew_Search(Crew_Search_Data):
    def __init__(self, ship):
        super(Two_Crew_Search, self).__init__()

class Crew_Probs: # contains all the prob of crew in each cell
    def __init__(self):
        self.bot_distance = 0 # distance of curr cell to bot, i.e, p(bi|cj)
        self.crew_prob = 0 # p(c)
        self.crew_given_beep = 0 # p(c|b)
        self.crew_given_no_beep = 0 # p(c|¬b)
        self.crew_and_beep = 0 # p (c,b)
        self.crew_and_no_beep = 0 # p(c,¬b)
        self.track_beep = list((0, 0))

class Alien_Probs:
    def __init__(self):
        self.alien_prob = 0
        self.alien_given_beep_obs = 0
        self.alien_and_beep = 0


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


""" Core Ship Layout (same as last project) """

class Ship:
    def __init__(self, size, log_level = LOG_INFO):
        self.size = size
        self.grid = [[Cell(i, j, CLOSED_CELL) for j in range(size)] for i in range(size)] # grid is a list of list with each cell as a class
        self.open_cells = []
        self.logger = Logger(log_level)
        self.isBeep = 0
        self.bot = (0, 0)
        self.alien = (0, 0)
        self.crew_1 = (0, 0)
        self.crew_2 = (0, 0)

        self.generate_grid()

    def get_cell(self, cord):
        return self.grid[cord[0]][cord[1]]

    def generate_grid(self):
        self.assign_start_cell()
        self.unblock_closed_cells()
        self.unblock_dead_ends()

    def assign_start_cell(self):
        random_row = randint(0, self.size - 1)
        random_col = randint(0, self.size - 1)
        self.grid[random_row][random_col].cell_type = OPEN_CELL
        self.open_cells.append((random_row, random_col))

    def unblock_closed_cells(self):
        available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)
        while available_cells:
            closed_cell = choice(available_cells)
            self.get_cell(closed_cell).cell_type = OPEN_CELL
            self.open_cells.append(closed_cell)
            available_cells = self.cells_with_one_open_neighbor(CLOSED_CELL)

    def unblock_dead_ends(self):
        dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)
        half_len = len(dead_end_cells)/2

        while half_len > 0:
            half_len -= 1
            dead_end_cell = choice(dead_end_cells)
            closed_neighbors = get_neighbors(
                self.size, dead_end_cell, self.grid, CLOSED_CELL
            )
            random_cell = choice(closed_neighbors)
            self.get_cell(random_cell).cell_type = OPEN_CELL
            self.open_cells.append(random_cell)
            if half_len:
                dead_end_cells = self.cells_with_one_open_neighbor(OPEN_CELL)

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
        self.get_cell(self.alien).cell_type = ALIEN_CELL

    def place_players(self):
        self.bot = choice(self.open_cells)
        self.open_cells.remove(self.bot)
        self.crew_1 = choice(self.open_cells)
        self.open_cells.remove(self.crew_1)
        self.crew_2 = choice(self.open_cells)
        self.open_cells.remove(self.crew_2)
        self.place_aliens()
        self.init_cell_details()
        self.set_player_cell_type()

    def init_cell_details(self):
        for i, cells in enumerate(self.grid):
            for j, cell in enumerate(cells):
                if cell.cell_type & (OPEN_CELL | BOT_CELL | CREW_CELL | ALIEN_CELL):
                    cell.listen_beep.crew_1_dist = get_manhattan_distance(self.crew_1, (i, j))
                    cell.listen_beep.crew_2_dist = get_manhattan_distance(self.crew_2, (i, j))
                    cell.listen_beep.c1_beep_prob = LOOKUP_E[cell.listen_beep.crew_1_dist]
                    cell.listen_beep.c2_beep_prob = LOOKUP_E[cell.listen_beep.crew_2_dist]
                    cell.cord = (i, j)

    def place_aliens(self):
        cells_within_zone = self.get_detection_zone(self.bot)
        self.initial_alien_cells = [cell_cord for cell_cord in self.open_cells if cell_cord not in cells_within_zone]

        while(True):
            self.alien = (randint(0, self.size -1), randint(0, self.size -1))
            if (self.alien not in self.initial_alien_cells):
                break

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

        return cells_within_zone

    def reset_detection_zone(self, curr_cell):
        cells_within_zone = self.get_detection_zone(curr_cell)

        # Reset cells outside detection zone to false
        for i, cells in enumerate(self.grid):
            for j, cell in enumerate(cells):
                if (i, j) not in cells_within_zone: cell.within_detection_zone = False

        return cells_within_zone

    def get_beep(self, cell, type):
        self.isBeep = uniform(0, 1)
        if (type == 1):
            return True if self.isBeep <= self.get_cell(cell).listen_beep.c1_beep_prob else False
        else:
            return True if self.isBeep <= self.get_cell(cell).listen_beep.c2_beep_prob else False

    def crew_beep(self, cell, crew_count, pending_crew):
        if crew_count == 2:
            if not pending_crew:
                c1_beep = self.get_beep(cell, 1)
                c2_beep = self.get_beep(cell, 2)
                return c1_beep or c2_beep
            elif pending_crew == 1:
                return self.get_beep(cell, 1)
            elif pending_crew == 2:
                return self.get_beep(cell, 2)
        return self.get_beep(cell, 1)

    def reset_grid(self):
        self.set_player_cell_type()
        for cell in self.open_cells:
            self.get_cell(cell).cell_type = OPEN_CELL


""" Basic search algorithm, and the parent class for our bot """

class SearchAlgo:
    def __init__(self, ship, log_level):
        self.ship = ship
        self.curr_pos = ship.bot
        self.alien_pos = ship.alien
        self.last_pos = ()
        self.logger = Logger(log_level)
        # Working on few issues, will fix it ASAP
        self.alien_calculation_disabled = True

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
            neighbors = get_neighbors(self.ship.size, current_cell, self.ship.grid, (OPEN_CELL | CREW_CELL | ALIEN_CELL))
            for neighbor in neighbors:
                if (neighbor not in visited_cells):
                    bfs_queue.append((neighbor, path_traversed + [neighbor]))

        return [] #God forbid, this should never happen


""" Main parent class for all our bots, contain most of the common bot logic """

class ParentBot(SearchAlgo):
    def __init__(self, ship, log_level):
        super(ParentBot, self).__init__(ship, log_level)
        self.alien_evasion_data = Alien_Evasion_Data(ship)
        self.crew_search_data = Crew_Search_Data()
        self.total_crew_count = 2
        self.traverse_path = []
        self.pred_crew_cells = []
        self.is_keep_moving = self.is_inital_calc_done = False
        self.recalc_pred_cells = True
        self.pending_crew = 0
        self.logger.print_grid(self.ship.grid)

    def update_cell_mov_vals(self, crew_probs, cell_cord):
        crew_probs.bot_distance = get_manhattan_distance(self.curr_pos, cell_cord)
        crew_probs.beep_given_crew = LOOKUP_E[crew_probs.bot_distance]
        crew_probs.no_beep_given_crew = LOOKUP_NOT_E[crew_probs.bot_distance]
        crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
        crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
        self.crew_search_data.beep_prob += crew_probs.crew_and_beep
        self.crew_search_data.no_beep_prob += crew_probs.crew_and_no_beep

    def handle_crew_beep(self):
        self.crew_search_data.is_beep_recv = self.ship.crew_beep(self.curr_pos, self.total_crew_count, self.pending_crew)
        track_beep = self.ship.get_cell(self.curr_pos).crew_probs.track_beep
        track_beep[0] += 1
        if self.crew_search_data.is_beep_recv:
            track_beep[1] += 1

    def is_rescued(self, total_iter, idle_steps):
        if (self.curr_pos == self.ship.crew_1) and self.pending_crew != 2:
            distance_1 = get_manhattan_distance(self.ship.crew_1, self.ship.bot)
            self.logger.print(LOG_INFO, f"Congrats, you found crew member 1 who was initially {distance_1} steps away from you after {total_iter} steps. You moved {total_iter - idle_steps} steps, and waited for {idle_steps} steps")
            self.pending_crew = 2
        elif (self.curr_pos == self.ship.crew_2) and self.pending_crew != 1:
            distance_2 = get_manhattan_distance(self.ship.crew_2, self.ship.bot)
            self.logger.print(LOG_INFO, f"Congrats, you found crew member 2 who was initially {distance_2} steps away from you after {total_iter} steps. You moved {total_iter - idle_steps} steps, and waited for {idle_steps} steps")
            self.pending_crew = 1
        return not self.total_crew_count

    def find_traverse_path(self):
        prob_crew_cell = ()
        if len(self.traverse_path) == 0 and len(self.pred_crew_cells) != 0:
            prob_crew_cell = self.pred_crew_cells.pop(0)
            self.traverse_path = self.search_path(prob_crew_cell)
            if len(self.traverse_path) == 0:
                self.logger.print(LOG_NONE, f"{self.traverse_path} {self.pred_crew_cells} {self.curr_pos} {prob_crew_cell}")
            self.logger.print(LOG_DEBUG, f"New path found, {self.traverse_path}. Pending cells to explore, {self.pred_crew_cells}")
            self.traverse_path.pop(0)

        if len(self.traverse_path) == 0: # some edge case handling
            self.logger.print(LOG_DEBUG, f"Bot in {self.curr_pos} with crew cells {self.crew_search_data.crew_cells} and last prob_crew_cell was {prob_crew_cell}")
            self.logger.print(LOG_DEBUG, f"Bot started {self.ship.bot} with crew at {self.ship.crew_1}")
            self.logger.print(LOG_DEBUG, f"pred_crew_cells::{self.pred_crew_cells}")
            return False

        return True


    """
        Ideally it is better to move the bot in the direction of the highest prob
        To do this, pred_crew_cells should be sorted based on probabilty
        Remember, we are not taking into account where the alien will be here!!
    """
    def move_bot(self):
        if not self.find_traverse_path():
            return False

        self.ship.get_cell(self.curr_pos).cell_type = OPEN_CELL
        self.last_pos = self.curr_pos
        self.curr_pos = self.traverse_path.pop(0)
        curr_cell = self.ship.get_cell(self.curr_pos)
        if (curr_cell.cell_type & CREW_CELL):
            curr_cell.cell_type |= BOT_CELL
            # curr_cell.crew_probs.crew_prob = 1
            self.total_crew_count -= 1
        # elif (curr_cell.cell_type & ALIEN_CELL):
        #     curr_cell.cell_type |= ALIEN_CELL         #  OOPS BYE BYE
        else:
            curr_cell.cell_type = BOT_CELL

        self.is_keep_moving = True if len(self.traverse_path) or len(self.pred_crew_cells) else False
        self.recalc_pred_cells = not self.is_keep_moving
        self.logger.print(LOG_DEBUG, f"Bot {self.last_pos} has moved to {self.curr_pos} with {self.total_crew_count} crew pending")
        return True

    def move_aliens(self):
        if self.alien_calculation_disabled: return
        alien_moves_possible = get_neighbors(
                self.ship.size,
                self.alien_pos,
                self.ship.grid,
                ALIEN_MOVEMENT_CELLS
            )

        if len(alien_moves_possible) == 0:
            return

        self.logger.print(
            LOG_DEBUG,
            f"Alien has moves {alien_moves_possible}"
        )

        alien_new_pos = choice(alien_moves_possible)

        old_alien_pos = self.alien_pos
        self.alien_pos = alien_new_pos

        if self.ship.grid[alien_new_pos] & BOT_CELL:
            self.logger.print(
                LOG_DEBUG,
                f"Alien moves from current cell {old_alien_pos} to bot cell {alien_new_pos}",
            )
            self.bot_caught_cell = alien_new_pos
            self.ship.grid[alien_new_pos[0]][alien_new_pos[1]].cell_type = BOT_CAUGHT_CELL
            return True

        else:
            self.logger.print(
                LOG_DEBUG,
                f"Alien moves from current cell {self.ship.grid[old_alien_pos]} to open cell {alien_new_pos}",
            )
            self.ship.grid[alien_new_pos[0]][alien_new_pos[1]].cell_type = ALIEN_CELL

        self.ship.grid[old_alien_pos[0]][old_alien_pos[1]].cell_type = OPEN_CELL

        return False


    def start_rescue(self):
        self.logger.print(LOG_NONE, "I am not nice")
        exit()

""" Bot 1 as per given specification """

class Bot_1(ParentBot):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_1, self).__init__(ship, log_level)
        self.crew_search_data = One_Crew_Search(ship)
        self.total_crew_count = 1
        self.pending_crew = 1
        self.ship.get_cell(self.ship.crew_2).cell_type = OPEN_CELL

    def calc_initial_search_data(self):
        if (self.is_inital_calc_done):
            return

        crew_cell_size = len(self.crew_search_data.crew_cells)
        for cell_cord in self.crew_search_data.crew_cells:
            cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
            cell.crew_probs.crew_prob = 1/crew_cell_size
            cell = self.update_cell_mov_vals(cell.crew_probs, cell_cord)


        alien_cell_size = len(self.alien_evasion_data.present_alien_cells)
        for cell_cord in self.alien_evasion_data.present_alien_cells:
            cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
            cell.alien_probs.alien_prob = 1/alien_cell_size

        self.is_inital_calc_done = True

    def update_pred_crew_cells(self, max_prob, pred_crew_cells):
        self.pred_crew_cells = list()
        max_prob_cells = pred_crew_cells[max_prob]
        curr_pos_beeps = self.ship.get_cell(self.curr_pos).crew_probs.track_beep
        curr_pos_fq = curr_pos_beeps[0]/curr_pos_beeps[1] if curr_pos_beeps[1] else 0
        if len(self.last_pos) and curr_pos_fq:
            last_pos_beeps = self.ship.get_cell(self.last_pos).crew_probs.track_beep
            last_pos_fq = last_pos_beeps[0]/last_pos_beeps[1] if last_pos_beeps[1] else 0
            max_prob_cells = sorted(max_prob_cells, key=lambda cell_pair: cell_pair[1], reverse=True)
            if curr_pos_fq > last_pos_fq:
                self.pred_crew_cells.append(max_prob_cells[-1][0])
            elif last_pos_fq > curr_pos_fq:
                self.pred_crew_cells.append(max_prob_cells[0][0])
            elif curr_pos_fq and last_pos_fq == curr_pos_fq:
                pos = round(len(max_prob_cells)/2)
                self.pred_crew_cells.append(max_prob_cells[pos][0])
        else: # can be randomized now...
            self.pred_crew_cells.append(choice(max_prob_cells)[0])
        self.logger.print(LOG_DEBUG, f"The new pred crew cells are {self.pred_crew_cells}")
        return

    def update_crew_probs_on_movement(self):
        self.crew_search_data.beep_prob = 0
        self.crew_search_data.no_beep_prob = 0
        for cell_cord in self.crew_search_data.crew_cells:
            cell = self.ship.get_cell(cell_cord)
            self.update_cell_mov_vals(cell.crew_probs, cell_cord)

    def check_nearby_crew(self):
        if self.crew_search_data.is_beep_recv:
            return False # no need to check nearby cells when we get a beep

        neighbors = get_neighbors(self.ship.size, self.curr_pos, self.ship.grid, (OPEN_CELL | CREW_CELL))
        neighbors_in_crew = [neighbor for neighbor in neighbors if neighbor in self.crew_search_data.crew_cells]
        if (len(neighbors_in_crew)):
            for neighbor in neighbors_in_crew:
                self.crew_search_data.crew_cells.remove(neighbor)
                if neighbor in self.pred_crew_cells:
                    self.pred_crew_cells.remove(neighbor)

                if self.is_keep_moving:
                    self.is_keep_moving = True if len(self.traverse_path) or len(self.pred_crew_cells) else False
                    self.recalc_pred_cells = not self.is_keep_moving

                cell = self.ship.get_cell(neighbor)
                cell.crew_probs.crew_prob = 0
                # self.crew_search_data.beep_prob -= cell.crew_probs.crew_and_beep
                # self.crew_search_data.no_beep_prob -= cell.crew_probs.crew_given_no_beep
            # recalc for all cells, normalizing our values is the same as, prob(a|¬b,¬c,¬d)=prob(a)/(1-p(¬b)-p(¬c)-p(¬d)))
            # will this work if i just remove the vals from beep_prob and go do the next calc right away??
            self.logger.print(LOG_DEBUG, f"Following cells {neighbors_in_crew}, were removed from crew cells {self.crew_search_data.crew_cells} and pred cells {self.pred_crew_cells}")
            return True
        return False

    def update_crew_search_data(self, bot_moved):
        beep_prob = no_beep_prob = 0
        pred_crew_cells = dict()
        max_prob = 0

        if (len(self.crew_search_data.crew_cells) == 0):
             self.logger.print(LOG_NONE, f"Bot in {self.curr_pos} has no crew cells!!!")
             self.logger.print(LOG_NONE, f"Bot started {self.ship.bot} with crew at {self.ship.crew_1}")
             self.logger.print(LOG_NONE, f"pred_crew_cells::{self.pred_crew_cells}")
             exit()

        if(self.check_nearby_crew() or bot_moved):
            self.update_crew_probs_on_movement()

        self.logger.print(LOG_DEBUG, f"is_beep_recv::{self.crew_search_data.is_beep_recv}")
        self.logger.print(LOG_DEBUG, f"beep_prob::{self.crew_search_data.beep_prob}, no_beep_prob::{self.crew_search_data.no_beep_prob}")
        for cell_cord in self.crew_search_data.crew_cells:
            cell = self.ship.get_cell(cell_cord)
            crew_probs = cell.crew_probs
            self.logger.print_cell_data(LOG_DEBUG, cell, self.curr_pos)
            if not (self.crew_search_data.beep_prob): # some serious issue
                self.logger.print_cell_data(LOG_NONE, cell, self.curr_pos)
                self.logger.print(LOG_NONE, f"is_beep_recv::{self.crew_search_data.is_beep_recv}")
                self.logger.print(LOG_NONE, f"beep_prob::{self.crew_search_data.beep_prob}, no_beep_prob::{self.crew_search_data.no_beep_prob}")
                self.logger.print(LOG_NONE, f"Bot in {self.curr_pos} has updated crew cells to be, {self.crew_search_data.crew_cells}. The pred cells is {self.pred_crew_cells}, with traverse path {self.traverse_path}")
                exit()

            crew_probs.crew_given_beep = (crew_probs.crew_and_beep) / self.crew_search_data.beep_prob
            if (self.crew_search_data.no_beep_prob != 0):
                crew_probs.crew_given_no_beep = (crew_probs.crew_and_no_beep) / self.crew_search_data.no_beep_prob

            crew_probs.crew_prob = crew_probs.crew_given_beep if self.crew_search_data.is_beep_recv else crew_probs.crew_given_no_beep
            if crew_probs.crew_prob not in pred_crew_cells:
                pred_crew_cells[crew_probs.crew_prob] = list()

            pred_crew_cells[crew_probs.crew_prob].append((cell_cord, crew_probs.bot_distance))
            if crew_probs.crew_prob > max_prob:
                max_prob = crew_probs.crew_prob

            crew_probs.crew_and_beep = crew_probs.beep_given_crew * crew_probs.crew_prob
            crew_probs.crew_and_no_beep = crew_probs.no_beep_given_crew * crew_probs.crew_prob
            beep_prob += crew_probs.crew_and_beep
            no_beep_prob += crew_probs.crew_and_no_beep
            self.logger.print_cell_data(LOG_DEBUG, cell, self.curr_pos)

        self.crew_search_data.beep_prob = beep_prob
        self.crew_search_data.no_beep_prob = no_beep_prob
        self.logger.print(LOG_DEBUG, f"beep_prob::{self.crew_search_data.beep_prob}, no_beep_prob::{self.crew_search_data.no_beep_prob}")
        self.logger.print(LOG_DEBUG, f"Bot in {self.curr_pos} has updated crew cells to be, {self.crew_search_data.crew_cells}.")
        self.logger.print_crew_probs(self.ship.grid, self.crew_search_data.is_beep_recv, self.curr_pos)
        return max_prob, pred_crew_cells

    def compute_likely_alien_movements(self):
        if self.alien_calculation_disabled: return
        alien_cells = self.alien_evasion_data.present_alien_cells # N at any point t
        prob_cell_mapping = dict()

        for cell_cord in alien_cells:
            cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
            prob_cell_mapping[cell_cord] = cell.alien_probs.alien_prob
            cell.alien_probs.alien_prob = 0

        self.logger.print(
            LOG_DEBUG, f"alien_cells::{alien_cells}, prob_cell_mapping::{prob_cell_mapping}"
        )

        for cell_cord in alien_cells:
            cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
            cell_prob = prob_cell_mapping[cell_cord]
            possible_alien_moves = get_neighbors(
                self.ship.size,
                cell_cord,
                self.ship.grid,
                ALIEN_MOVEMENT_CELLS
            )

            total_moves = len(possible_alien_moves)

            if(total_moves == 0):
                cell.alien_probs.alien_prob = cell_prob
                continue

            for alien_move in possible_alien_moves:
                new_cell = self.ship.grid[alien_move[0]][alien_move[1]]
                new_cell.alien_probs.alien_prob += cell_prob/total_moves
                self.logger.print(
                    LOG_DEBUG, f"alien_cells::{alien_cells}, prob_cell_mapping::{prob_cell_mapping}"
                )

        prob_cell_mapping.clear()

        if self.logger.log_level == LOG_DEBUG:
            for cell_cord in alien_cells:
                cell = self.ship.grid[cell_cord[0]][cell_cord[1]]
                self.logger.print(LOG_DEBUG, f"Now the cell.alien_prob for cell_cord::{cell_cord} is {cell.alien_probs.alien_prob}")

    '''
        If beep heard, cells within detection zone: P(B obs /A) = 1
        cells outside P(B obs /A) = 0
        If beep not heard, cells within detection zone: P(B obs /A) = 0
        cells outside P(B obs /A) = 1
        P(A/B obs) = P(B obs/ A) P(A) / P(B obs)
    '''
    def calc_alien_probs(self):
        if self.alien_calculation_disabled: return
        alien_cells = self.alien_evasion_data.all_alien_cells
        beep_recv = self.alien_evasion_data.is_beep_recv

        prob_alien_in_inner_cells = ALIEN_PRESENT if beep_recv else ALIEN_NOT_PRESENT
        prob_alien_in_outer_cells = ALIEN_PRESENT if (not beep_recv) else ALIEN_NOT_PRESENT

        for cell_cord in alien_cells:
            cell = self.ship.grid[cell_cord[0]][cell_cord[1]]

            prob_beep_gv_alien = prob_alien_in_inner_cells if cell.within_detection_zone else prob_alien_in_outer_cells
            cell.alien_probs.alien_and_beep = prob_beep_gv_alien * cell.probs.alien_prob

            # cell.probs.alien_given_beep_obs =
            continue
        pass

    def rescue_info(self):
        init_1_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_1_dist
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_1}")
        return init_1_distance

    def update_crew_cells(self):
        if self.curr_pos in self.crew_search_data.crew_cells:
            self.crew_search_data.crew_cells.remove(self.curr_pos)
            self.logger.print(LOG_DEBUG, f"Removing current position{self.curr_pos} from list of probable crew cells{self.crew_search_data.crew_cells}")
            self.crew_search_data.beep_prob -= self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].crew_probs.crew_and_beep
            self.crew_search_data.no_beep_prob -= self.ship.grid[self.curr_pos[0]][self.curr_pos[1]].crew_probs.crew_and_no_beep

    def start_rescue(self):
        bot_moved = is_move_bot = False
        beep_counter = idle_steps = total_iter = 0
        init_distance = self.rescue_info()

        while (True): # Keep trying till you find the crew
            total_iter += 1
            self.calc_initial_search_data()

            # Alien beep logic
            alien_cell = self.ship.grid[self.alien_pos[0]][self.alien_pos[1]]
            self.alien_evasion_data.is_beep_recv = alien_cell.within_detection_zone
            self.logger.print(
                LOG_DEBUG,
                f"alien_pos:{self.alien_pos}, within_detection_zone::{alien_cell.within_detection_zone}"
            )

            self.handle_crew_beep()
            max_prob, pred_crew_cells = self.update_crew_search_data(bot_moved)
            self.calc_alien_probs()
            if self.recalc_pred_cells:
                self.update_pred_crew_cells(max_prob, pred_crew_cells)

            if (self.is_keep_moving or is_move_bot) and self.move_bot():
                if self.is_rescued(total_iter, idle_steps):
                    return (init_distance, total_iter, idle_steps)

                self.update_crew_cells()
                beep_counter = 0
                is_move_bot = False
                bot_moved = True
                self.ship.reset_detection_zone(self.curr_pos)
            else:
                is_move_bot = False
                bot_moved = False
                idle_steps += 1

            if self.move_aliens():
                return

            # update probability of alien movement based on current P(A), cell.alien_prob
            self.compute_likely_alien_movements()

            beep_counter += 1
            if beep_counter >= (IDLE_BEEP_COUNT - 1):
                is_move_bot = True

class Bot_3(Bot_1):
    def __init__(self, ship, log_level = LOG_NONE):
        super(Bot_3, self).__init__(ship, log_level)
        self.total_crew_count = 2
        self.pending_crew = 0
        self.ship.get_cell(self.ship.crew_2).cell_type = CREW_CELL

    def rescue_info(self):
        init_1_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_1_dist
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_1}")
        init_2_distance = self.ship.get_cell(self.curr_pos).listen_beep.crew_2_dist
        self.logger.print(LOG_INFO, f"Bot{self.curr_pos} needs to find the crew{self.ship.crew_2}")
        return init_1_distance + self.ship.get_cell(self.ship.crew_1).listen_beep.crew_2_dist

"""Simulation & Testing logic begins"""

# Responsible for updating the alpha for each worker pool
def update_lookup(alpha):
    global LOOKUP_E, LOOKUP_NOT_E, ALPHA
    ALPHA = alpha
    LOOKUP_E = [pow(exp, (-1*ALPHA*(i - 1))) for i in range(GRID_SIZE*2 + 1)]
    LOOKUP_NOT_E = [(1-LOOKUP_E[i]) for i in range(GRID_SIZE*2 + 1)]

# Test function
def run_test(log_level = LOG_INFO):
    update_lookup(ALPHA)
    ship = Ship(GRID_SIZE)
    ship.place_players()
    bot_1 = Bot_1(ship, log_level)
    bot_1.start_rescue()
    del bot_1
    del ship

def bot_factory(itr, ship):
    if (itr % TOTAL_BOTS == 0):
        return Bot_1(ship)
    elif (itr % TOTAL_BOTS == 1):
        return Bot_3(ship)
    return ParentBot(ship, LOG_NONE)

# Runs n number of iteration for each bot for given alpha value
def run_sim(my_range, queue, alpha):
    update_lookup(alpha)
    temp_data_set = [[0.0 for i in range(4)] for j in range(TOTAL_BOTS)]
    space_itr = round((my_range[0]/100) + 1)
    for itr in my_range:
        print(itr+1, end='\r')
        ship = Ship(GRID_SIZE)
        ship.place_players()
        for i in range(TOTAL_BOTS):
            bot = bot_factory(i, ship)
            begin = time()
            ret_vals = bot.start_rescue()
            end = time()
            for j in range(3):
                temp_data_set[i][j] += ret_vals[j]
            temp_data_set[i][3] += (end-begin)
            ship.reset_grid()
            del bot
        del ship

    queue.put(temp_data_set)

# Creates "n" process, and runs multiple simulation simultaneously
def run_multi_sim(alpha, is_print = False):
    begin = time()
    data_set = [[0.0 for i in range(4)] for j in range(TOTAL_BOTS)]
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
        for i in range(TOTAL_BOTS):
            for j in range(4):
                data_set[i][j] += temp_data_set[i][j]

    for i in range(TOTAL_BOTS):
        data_set[i][0] = data_set[i][0]/actual_iters
        data_set[i][1] = data_set[i][1]/actual_iters
        data_set[i][2] = data_set[i][2]/actual_iters
        data_set[i][3] = data_set[i][3]/actual_iters
    end = time()
    if (is_print):
        print()
        print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {alpha}")
        print ("%20s %20s %20s %20s %20s" % ( "Distance", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
        for i in range(TOTAL_BOTS):
            print("%20s %20s %20s %20s %20s" % (data_set[i][0], data_set[i][1], data_set[i][2], data_set[i][1] - data_set[i][2], data_set[i][3]))
    else:
        print(f"Grid Size:{GRID_SIZE} for {actual_iters} iterations took time {end-begin} for alpha {alpha}")

    del queue
    del processes
    return data_set[0]

# Runs multiple simulations for multiple values of alpha
def compare_multiple_alpha():
    global ALPHA
    begin = time()
    alpha_map = {}
    for itr in range(MAX_ALPHA_ITERATIONS):
        alpha_map[ALPHA] = run_multi_sim(ALPHA)
        ALPHA = round(ALPHA + ALPHA_STEP_INCREASE, 2)
    end = time()
    print(f"It took {end - begin} seconds to complete computation")
    print ("%20s %20s %20s %20s %20s %20s" % ("Alpha", "Distance", "Total Iterations", "Idle steps", "Steps moved", "Time taken"))
    for key, value in alpha_map.items():
        print("%20s %20s %20s %20s %20s %20s" % (key, value[0], value[1], value[2], value[1]-value[2], value[3]))

# BOT HAS SOME LOGIC IN PROB FOR SOME REASON!!!!
if __name__ == '__main__':
    # run_test()
    run_multi_sim(ALPHA, True)
    # compare_multiple_alpha()
