import AI_Proj3
import AI_Bonus3
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from time import time
from math import ceil
import csv
import os

AI_Proj3.GRID_SIZE = 7
AI_Proj3.NO_CLOSED_CELLS = False
AI_Proj3.RAND_CLOSED_CELLS = 0
AI_Proj3.CONVERGENCE_LIMIT=0.01 # Small value to reduce time complexity

TOTAL_ITERATIONS = 100 # iterations for same ship layout and different bot/crew positions
IS_BONUS = True
TOTAL_CONFIGS = 1 if IS_BONUS else 2
MAX_CORES = cpu_count()

class DETAILS:
    def __init__(self):
        self.success = self.failure = self.caught = 0.0
        self.s_moves = self.f_moves = self.c_moves = 0.0
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
        self.caught += new_detail.caught
        self.c_moves += new_detail.c_moves
        self.distance += new_detail.distance
        self.dest_dist += new_detail.dest_dist
        self.update_min_max(new_detail.max_success)
        self.update_min_max(new_detail.min_success)

    def get_avg(self, total_itr):
        if self.success:
            self.s_moves /= self.success

        if self.failure:
            self.f_moves /= self.failure

        if self.caught:
            self.c_moves /= self.caught

        self.success /= total_itr
        self.failure /= total_itr
        self.caught /= total_itr
        self.distance /= total_itr
        self.dest_dist /= total_itr

def bot_fac(itr, myship):
    if IS_BONUS:
        return AI_Bonus3.ALIEN_CONFIG(myship)

    if itr % TOTAL_CONFIGS  == 0:
        return AI_Proj3.NO_BOT_CONFIG(myship)
    else:
        return AI_Proj3.BOT_CONFIG(myship)

def run_sim(args):
    if len(args) == 1:
        ship = AI_Bonus3.ALIEN_SHIP() if IS_BONUS else AI_Proj3.SHIP()
        ship.perform_initial_calcs()
    else:
        ship = args[1]

    avg_moves = [DETAILS() for itr in range(TOTAL_CONFIGS)]
    for _ in args[0]:
        # print(_, end = "\r")
        dest_dist = AI_Proj3.get_manhattan_distance(ship.crew_pos, ship.teleport_cell)
        for itr in range(TOTAL_CONFIGS):
            test_bot = bot_fac(itr, ship)
            moves, result = test_bot.start_rescue()
            ship.reset_grid()
            if result:
                avg_moves[itr].update_min_max(moves)
                avg_moves[itr].s_moves += moves
                avg_moves[itr].success += 1
            elif result == 2:
                avg_moves[itr].c_moves += moves
                avg_moves[itr].caught += 1
            else:
                avg_moves[itr].f_moves += moves
                avg_moves[itr].failure += 1

            distance = 0 if test_bot.__class__ is AI_Proj3.NO_BOT_CONFIG else AI_Proj3.get_manhattan_distance(ship.bot_pos, ship.crew_pos)
            avg_moves[itr].distance += distance
            avg_moves[itr].dest_dist += dest_dist
            del test_bot

        ship.reset_positions()

    # print()
    del ship
    return avg_moves

def print_header(total_itr = TOTAL_ITERATIONS):
    print("Total iterations performed for layout is", total_itr)
    print("%3s %18s %18s %18s %18s %18s %18s %18s %18s %18s %18s" % ("No", "Avg Suc Moves", "Success Rate", "Min Suc. Moves", "Max Suc. Moves", "Avg Caught Moves", "Caught Rate", "Avg Fail Moves", "Failure Rate", "Avg Bot Crew Dist", "Crew Teleport Dist"))

def print_data(final_data, itr, total_itr = TOTAL_ITERATIONS):
    final_data[itr].get_avg(total_itr)
    print(("%3s %18s %18s %18s %18s %18s %18s %18s %18s %18s %18s" % (itr, final_data[itr].s_moves, final_data[itr].success, final_data[itr].min_success, final_data[itr].max_success, final_data[itr].c_moves, final_data[itr].caught, final_data[itr].f_moves, final_data[itr].failure, final_data[itr].distance, final_data[itr].dest_dist)))

def run_multi_sim():
    core_count = MAX_CORES
    arg_data = [[range(0, TOTAL_ITERATIONS)] for i in range(core_count)]
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
    final_data = run_sim([range(0, total_itr)])

    print_header(total_itr)
    for itr in range(TOTAL_CONFIGS):
        print_data(final_data, itr, total_itr)

def single_run():
    ship = ALIEN_SHIP() if IS_BONUS else SHIP()
    ship.perform_initial_calcs()
    ship.print_ship()
    for itr in range(TOTAL_CONFIGS):
        test_bot = bot_fac(itr, ship)
        print(test_bot.start_rescue())
        ship.reset_grid()

def create_file(file_name):
    column_headings = ["State", "Bot_Pos", "Bot_Moves", "State_Values", "Best_Move"]
    # Check if the file exists and is empty
    if not os.path.isfile(file_name) or os.stat(file_name).st_size == 0:
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_headings)

def generate_data(args):
    file_name = args[0]
    sim_range = args[1]
    for _ in sim_range:
        ship = ALIEN_SHIP() if IS_BONUS else SHIP()
        ship.perform_initial_calcs()
        bot_config = BOT_CONFIG(ship)
        bot_config.start_data_collection(file_name)
        del bot_config
        del ship

def get_generalized_data():
    core_count = MAX_CORES
    total_data = 100
    thread_data = ceil(total_data/core_count)
    arg_data = [("output_"+str(i)+".csv", range(0, thread_data)) for i in range(core_count)]
    final_out = "output.csv"
    create_file(final_out)
    with open(final_out, 'a', newline='') as csvfile:
        with Pool(processes=core_count) as p:
            p.map(generate_data, arg_data)
            for args in arg_data:
                file_name = args[0]
                with open(file_name, mode ='r') as read_file:
                    csv_file = csv.reader(read_file)
                    for lines in csv_file:
                        writer = csv.writer(csvfile)
                        writer.writerow(lines)
                os.remove(file_name)

def generate_same_data(args):
    file_name = args[0]
    sim_range = args[1]
    ship = deepcopy(args[2])
    for _ in sim_range:
        bot_config = AI_Proj3.BOT_CONFIG(ship)
        bot_config.start_data_collection(file_name)
        del bot_config
        ship.reset_positions()

    del ship

def get_single_data():
    core_count = MAX_CORES
    total_data = 10000
    thread_data = ceil(total_data/core_count)
    ship = AI_Bonus3.ALIEN_SHIP() if IS_BONUS else AI_Proj3.SHIP()
    ship.perform_initial_calcs()
    arg_data = [("single_"+str(i)+".csv", range(0, thread_data), ship) for i in range(core_count)]
    final_out = "single.csv"
    create_file(final_out)
    with open(final_out, 'a', newline='') as csvfile:
        with Pool(processes=core_count) as p:
            p.map(generate_same_data, arg_data)
            for args in arg_data:
                file_name = args[0]
                with open(file_name, mode ='r') as read_file:
                    csv_file = csv.reader(read_file)
                    for lines in csv_file:
                        writer = csv.writer(csvfile)
                        writer.writerow(lines)
                os.remove(file_name)

    del ship

if __name__ == '__main__':
    begin = time()
    # single_run()
    single_sim(1000)
    # run_multi_sim()
    # get_single_data()
    # get_generalized_data()
    end = time()
    print(end-begin)
