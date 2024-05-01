import AI_Proj3
from run_simulations import DETAILS
import torch
from matplotlib import pylab as plt
from torch import nn
import torch.nn.functional as F
from time import time
import numpy as np
import random

MAX_TEST = 1000
MAX_TRAIN = 1000

AI_Proj3.GRID_SIZE = 11
FULL_GRID_STATE = AI_Proj3.TOTAL_ELEMENTS*AI_Proj3.GRID_SIZE**2
# FULL_GRID_STATE = AI_Proj3.GRID_SIZE**2
H_LAYERS = [750, 1000, 750]
BOT_ACTIONS = 9

ACTIONS_ID = {
"IDLE" : int(0),
"NORTH" : int(1),
"SOUTH" : int(5),
"EAST" : int(3),
"WEST" : int(7),
"NORTH_EAST" : int(2),
"NORTH_WEST" : int(8),
"SOUTH_EAST" : int(4),
"SOUTH_WEST" : int(6)
}

class HIDDEN_UNITS(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(HIDDEN_UNITS, self).__init__()
        self.activation = activation
        self.nn = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.nn(x)
        out = self.activation(out)
        return out

class QModel(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, activation = F.relu):
        super(QModel, self).__init__()
        assert type(hidden_layers) is list
        self.hidden_units = nn.ModuleList()
        self.in_channels = in_channels
        prev_layer = in_channels
        for hidden in hidden_layers:
            self.hidden_units.append(HIDDEN_UNITS(prev_layer, hidden, activation))
            prev_layer = hidden
        self.final_unit = nn.Linear(prev_layer, out_channels)

    def forward(self, x):
        out = x.view(-1,self.in_channels).float()
        for unit in self.hidden_units:
            out = unit(out)
        out = self.final_unit(out)
        return out

class LEARN_CONFIG(AI_Proj3.SHIP):
    def __init__(self):
        super(LEARN_CONFIG, self).__init__()
        self.q_model = QModel(FULL_GRID_STATE, H_LAYERS, BOT_ACTIONS)
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=AI_Proj3.CONVERGENCE_LIMIT)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.losses = []
        self.total_failure_moves = self.total_success_moves = 0

    def print_losses(self):
        # plt.figure(figsize=(10,7))
        # plt.plot(self.losses)
        # plt.xlabel("Epochs",fontsize=22)
        # plt.ylabel("Loss",fontsize=22)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        print(self.total_failure_moves, self.total_success_moves, self.total_failure_moves/(self.total_failure_moves+self.total_success_moves), len(self.losses))
        self.total_failure_moves = self.total_success_moves = 0
        self.losses.clear()

    def get_action(self, bot_cell, bot_move, is_parse = False):
        val = -1
        if bot_cell and bot_move:
            delta_x = bot_move[0] - bot_cell[0]
            delta_y = bot_move[1] - bot_cell[1]

            if delta_x == 0 and delta_y == 0:
                val = ACTIONS_ID["IDLE"]
            elif delta_x == 0:
                val = ACTIONS_ID["NORTH"] if delta_y > 0 else ACTIONS_ID["SOUTH"]
            elif delta_y == 0:
                val = ACTIONS_ID["EAST"] if delta_x > 0 else ACTIONS_ID["WEST"]
            elif delta_x > 0:
                val = ACTIONS_ID["NORTH_EAST"] if delta_y > 0 else ACTIONS_ID["SOUTH_EAST"]
            else:
                val = ACTIONS_ID["NORTH_WEST"] if delta_y > 0 else ACTIONS_ID["SOUTH_WEST"]
        return val


class Q_BOT(AI_Proj3.BOT_CONFIG):
    def __init__(self, ship, epsilon):
        super(Q_BOT, self).__init__(ship)
        self.old_bot_pos = ()
        self.old_crew_pos = ()
        self.state_1 = self.get_curr_state()
        self.tensor_1 = torch.from_numpy(self.state_1).float()
        self.state_2 = np.array([])
        self.is_train = True
        self.epsilon = epsilon
        self.legal_moves = []

    def get_curr_state(self):
        rows = []
        for i in range(self.ship.size):
            cols = []
            for j in range(self.ship.size):
                curr_state = self.ship.get_state((i, j))
                states = []
                states.append(-2 if curr_state == AI_Proj3.CLOSED_CELL else 0)
                states.append(5 if curr_state == AI_Proj3.TELEPORT_CELL else 0)
                states.append(1 if curr_state == AI_Proj3.BOT_CELL else 0)
                states.append(2 if curr_state == AI_Proj3.CREW_CELL else 0)
                # states.append(1 if curr_state == AI_Proj3.OPEN_CELL else 0)
                cols.extend(states)

            rows.extend(cols)

        # final = (np.asarray(rows, dtype=np.float64) + np.random.rand(1, FULL_GRID_STATE)/BOT_ACTIONS).flatten()
        final = (np.asarray(rows, dtype=np.float64))
        return final

    def make_action(self):
        move = self.local_all_moves[self.action_no]
        next_pos = (self.local_bot_pos[0] + move[0], self.local_bot_pos[1] + move[1])
        self.action_result = AI_Proj3.CLOSED_CELL
        self.legal_moves.clear()
        for itr, check_move in enumerate(self.local_all_moves):
            check_pos = (self.local_bot_pos[0] + check_move[0], self.local_bot_pos[1] + check_move[1])
            if 0 < check_pos[0] < self.ship.size and 0 < check_pos[1] < self.ship.size:
                curr_state = self.ship.get_state(check_pos)
                if curr_state != AI_Proj3.CLOSED_CELL and curr_state != AI_Proj3.CREW_CELL:
                    self.legal_moves.append(itr)

        # print(self.action_no, self.best_action, self.illegal_moves)
        if 0 < next_pos[0] < self.ship.size and 0 < next_pos[1] < self.ship.size:
            state = self.ship.get_state(next_pos)
            self.action_result = state
            if state != AI_Proj3.CLOSED_CELL and state != AI_Proj3.CREW_CELL:
                self.make_bot_move(next_pos)
                return


    def calc_loss(self):
        bot_pos = self.local_bot_pos
        crew_pos = self.local_crew_pos
        self.state_2 = self.get_curr_state()
        self.tensor_2 = torch.from_numpy(self.state_2).float()
        with torch.no_grad():
            possibleQs = self.ship.q_model(self.tensor_2)

        # newQ = GAAMA*possibleQs[action_no].item() - reward if self.local_crew_pos != self.ship.teleport_cell else 0
        # self.policy_reward = torch.Tensor([newQ]).detach()
        # self.policy_action = self.q_vals.squeeze()[self.action_no]
        # print(self.policy_action, self.policy_reward)
        # loss = self.ship.loss_fn(self.policy_action, self.policy_reward)

        action_list = [0.0]*BOT_ACTIONS
        action_list[self.best_action] = 1.0
        # move_prob = 0.1/(len(self.legal_moves) - 1) if (len(self.legal_moves) - 1) else 0.0
        # for pos in self.legal_moves:
        #     action_list[pos] = 0.9 if pos == self.best_action else move_prob

        if self.action_no != self.best_action:
            self.ship.total_failure_moves += 1
        else:
            self.ship.total_success_moves += 1

        # loss = self.ship.loss_fn(self.q_vals, torch.Tensor(action_list))
        loss = self.ship.loss_fn(self.q_vals, torch.Tensor([self.best_action]).long())
        if (self.is_train):
            self.ship.optimizer.zero_grad()
            loss.backward()
            self.ship.losses.append(loss.item())
            self.ship.optimizer.step()
        else:
            loss.backward()
            self.ship.losses.append(loss.item())

        self.state_1 = self.state_2
        self.tensor_1 = self.tensor_2

    def process_q_learn(self):
        self.q_vals = self.ship.q_model(self.tensor_1)
        self.best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_crew_pos][AI_Proj3.BEST_MOVE]
        self.best_action = self.ship.get_action(self.local_bot_pos, self.best_move)
        if random.uniform(0, 1) < self.epsilon:
            self.action_no = np.random.randint(0,9)
        else:
            self.action_no = np.argmax(self.q_vals.data.numpy())

        self.make_action()
        return self.move_crew()

    def move_bot(self):
        self.q_vals = self.ship.q_model(self.tensor_1)
        self.action_no = np.argmax(self.q_vals.data.numpy())
        # self.policy_action = self.q_vals.squeeze()[self.action_no]
        self.best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_crew_pos][AI_Proj3.BEST_MOVE]
        self.best_action = self.ship.get_action(self.local_bot_pos, self.best_move)
        self.make_action()

    def move_crew(self):
        ret_val = super(Q_BOT, self).move_crew()
        self.calc_loss()
        return ret_val

    def train_rescue(self):
        self.is_train = True
        self.total_moves = 0
        if self.ship.get_state(self.local_crew_pos) & AI_Proj3.TELEPORT_CELL:
            return self.total_moves, AI_Proj3.SUCCESS

        while(True):
            self.total_moves += 1
            # self.visualize_grid(False)
            if self.process_q_learn():
                # self.visualize_grid()
                return self.total_moves, AI_Proj3.SUCCESS

            if self.total_moves > 1000:
                # self.visualize_grid()
                return self.total_moves, AI_Proj3.FAILURE

    def test_rescue(self):
        self.is_train = False
        return self.start_rescue()

def t_bot(ship, is_train = True):
    epochs = MAX_TRAIN if is_train else MAX_TEST
    avg_moves = DETAILS()
    epsilon = 1.0
    for iters in range(epochs):
        q_bot = Q_BOT(ship, epsilon)
        if is_train:
            moves, result = q_bot.train_rescue()
        else:
            moves, result = q_bot.test_rescue()

        if result:
            avg_moves.update_min_max(moves)
            avg_moves.s_moves += moves
            avg_moves.success += 1
        else:
            avg_moves.f_moves += moves
            avg_moves.failure += 1

        avg_moves.distance += AI_Proj3.get_manhattan_distance(ship.bot_pos, ship.crew_pos)
        avg_moves.dest_dist += AI_Proj3.get_manhattan_distance(ship.crew_pos, ship.teleport_cell)
        del q_bot

        if epsilon > 0.1:
            epsilon -= (1/epochs)

        ship.reset_positions()

    print()
    avg_moves.get_avg(epochs)
    print(("%18s %18s %18s %18s %18s %18s %18s %18s" % (avg_moves.s_moves, avg_moves.success, avg_moves.min_success, avg_moves.max_success, avg_moves.f_moves, avg_moves.failure, avg_moves.distance, avg_moves.dest_dist)))

def single_sim(ship):
    final_data = AI_Proj3.run_sim([range(0, MAX_TEST), ship])

    AI_Proj3.print_header(MAX_TEST)
    for itr in range(AI_Proj3.TOTAL_CONFIGS):
        AI_Proj3.print_data(final_data, itr, MAX_TEST)

def single_run():
    ship = LEARN_CONFIG()
    ship.perform_initial_calcs()
    t_bot(ship)
    ship.print_losses()
    t_bot(ship, False)
    ship.print_losses()
    single_sim(ship)

if __name__ == '__main__':
    begin = time()
    single_run()
    end = time()
    print(end-begin)
