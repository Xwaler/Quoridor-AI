import numpy as np
import time
import os
import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Linear, Module, BCELoss, Sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from collections import deque
import time
from astar import astar
from copy import deepcopy
import random

BOARD_SIZE = 9


class Cell:
    def __init__(self, position):
        self.x, self.y = position
        self.player = None
        self.f_left, self.f_right, self.f_up, self.f_down = False, False, False, False

    def __repr__(self):
        rep = ''
        if self.f_left:
            rep += '<'
        if self.f_up:
            rep += '^'
        if self.f_down:
            rep += 'v'
        if self.f_right:
            rep += '>'
        if self.player:
            rep = f'[{rep}]' if self.player.no == 0 else f'({rep})'
        if not rep:
            rep = 'oo'
        return rep


class Player:
    def __init__(self, position, no):
        self.x, self.y = position
        self.no = no
        self.remaining_fences = 10
        self.objective_y = 0

    def won(self):
        return self.y == self.objective_y

    def __repr__(self):
        return f'Player {self.no}, ({self.y}, {self.x})'


class TopPlayer(Player):
    def __init__(self, position, no):
        super().__init__(position, no)
        self.objective_y = 8


class BottomPlayer(Player):
    def __init__(self, position, no):
        super().__init__(position, no)
        self.objective_y = 0


class Quoridor:
    def __init__(self):
        self.over = False
        self.cells = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=Cell)
        self.players = np.empty(2, dtype=Player)
        self.reset()

    def reset(self):
        self.over = False

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                self.cells[y][x] = Cell((x, y))

        self.players[:] = (TopPlayer((4, 0), 0), BottomPlayer((4, 8), 1))
        for player in self.players:
            self.cells[player.y][player.x].player = player

        return self.get_state(self.players[0])

    def get_random_action(self, player_id):
        player = self.players[player_id]

        valid = False
        action = None
        while not valid:
            if np.random.randint(2):
                action = [
                    0,
                    np.random.randint(4)
                ]
            else:
                x, y = np.random.randint(0, BOARD_SIZE, 2)
                action = [
                    1,
                    x,
                    y,
                    np.random.randint(2)
                ]

            valid = self.action_is_valid(player, action)
        return action

    def action_is_valid(self, player, action):
        # move : action = [0, direction]
        if action[0] == 0:
            _, direction = action
            player_cell = self.cells[player.y][player.x]
            return (direction == 0 and not player_cell.f_left and player.x - 1 >= 0) or \
                   (direction == 1 and not player_cell.f_up and player.y - 1 >= 0) or \
                   (direction == 2 and not player_cell.f_right and player.x + 1 < BOARD_SIZE) or \
                   (direction == 3 and not player_cell.f_down and player.y + 1 < BOARD_SIZE)

        # place fence : action = [1, x, y, vertical]
        else:
            _, origin_x, origin_y, vertical = action
            return player.remaining_fences > 0 and (
                    (vertical and origin_x > 0 and origin_y + 1 < BOARD_SIZE and
                        not self.cells[origin_y][origin_x].f_left and
                        not self.cells[origin_y + 1][origin_x].f_left)
                    or (not vertical and origin_y > 0 and origin_x + 1 < BOARD_SIZE and
                        not self.cells[origin_y][origin_x].f_up and
                        not self.cells[origin_y][origin_x + 1].f_up)
            ) and not self.is_blocking(action)

    def is_blocking(self, action):
        cells = self.get_sim_place_fence(action)
        paths_exist = self.paths_exists(cells)
        return not paths_exist

    def paths_exists(self, cells):
        for player in self.players:
            found = False
            start = (player.y, player.x)
            x = 0
            while not found and x < BOARD_SIZE:
                end = (player.objective_y, x)
                path = astar(cells, start, end)
                if path:
                    found = True
                else:
                    x += 1
            if not found:
                return False
        return True

    def get_sim_place_fence(self, action):
        _, origin_x, origin_y, vertical = action
        cells = deepcopy(self.cells)

        if vertical:
            cells[origin_y][origin_x].f_left = True
            cells[origin_y][origin_x - 1].f_right = True
            cells[origin_y + 1][origin_x].f_left = True
            cells[origin_y + 1][origin_x - 1].f_right = True

        else:
            cells[origin_y][origin_x].f_up = True
            cells[origin_y - 1][origin_x].f_down = True
            cells[origin_y][origin_x + 1].f_up = True
            cells[origin_y - 1][origin_x + 1].f_down = True

        return cells

    def place_fence(self, player, action):
        _, origin_x, origin_y, vertical = action

        if vertical:
            self.cells[origin_y][origin_x].f_left = True
            self.cells[origin_y][origin_x - 1].f_right = True
            self.cells[origin_y + 1][origin_x].f_left = True
            self.cells[origin_y + 1][origin_x - 1].f_right = True

        else:
            self.cells[origin_y][origin_x].f_up = True
            self.cells[origin_y - 1][origin_x].f_down = True
            self.cells[origin_y][origin_x + 1].f_up = True
            self.cells[origin_y - 1][origin_x + 1].f_down = True

        player.remaining_fences -= 1

    # left, up, right, down
    def move(self, player, action):
        _, direction = action
        player_cell = self.cells[player.y][player.x]

        if direction == 0:
            target = self.cells[player.y][player.x - 1]
        elif direction == 1:
            target = self.cells[player.y - 1][player.x]
        elif direction == 2:
            target = self.cells[player.y][player.x + 1]
        else:
            target = self.cells[player.y + 1][player.x]

        player_cell.player = None
        target.player = player
        player.x, player.y = target.x, target.y

    def step(self, player_id, action):
        player = self.players[player_id]

        if action[0] == 0:
            self.move(player, action)
        else:
            self.place_fence(player, action)

        if player.won():
            self.over = True

        return self.get_state(player), self.over

    def get_state(self, player):
        state = np.zeros((BOARD_SIZE, BOARD_SIZE, 6), dtype=np.bool)
        for j in range(BOARD_SIZE):
            for i in range(BOARD_SIZE):
                cell = self.cells[j][i]
                state[j][i] = [
                    cell.f_left,
                    cell.f_up,
                    cell.f_right,
                    cell.f_down,
                    cell.player is not None,
                    cell.player is not None and cell.player == player
                ]
        # if isinstance(player, TopPlayer):
        #     state = state[::-1, ::-1]

        return state.reshape((6, BOARD_SIZE, BOARD_SIZE))


def action_to_pred(action):
    pred = np.random.randint(0, 2, 21, dtype=np.bool)
    pred[0] = action[0]
    if action[0] == 0:
        pred[1:5] = action[1:5]
    else:
        pred[1:21] = np.zeros(20, dtype=np.bool)
        pred[1 + action[1]] = 1
        pred[10 + action[2]] = 1
        pred[19 + action[3]] = 1
    return pred


def pred_to_action(pred):
    action = [int(round(pred[0]))]
    if action[0] == 0:
        action.append(np.argmax(pred[1:5]))
    else:
        action.append(np.argmax(pred[1:10]))
        action.append(np.argmax(pred[10:19]))
        action.append(np.argmax(pred[19:21]))
    return action


"""
    output: (21,)
    [0, left, up right, down, ...]
    [1, x=0...x=8, y=0...y=8, h, v]
"""


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=6, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.linear = Linear(256 * 4 * 4, 21)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 4 * 4)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    game = Quoridor()

    MIN_HISTORY = 5_000
    MAX_HISTORY = 20_000
    history = deque(maxlen=MAX_HISTORY)

    LOSS_HISTORY = 100
    losses = deque(maxlen=LOSS_HISTORY)

    BATCH_SIZE = 32
    LR = 0.001
    net = Net().to(device)
    criterion = BCELoss()
    optimizer = Adam(net.parameters(), lr=LR)

    k = 0
    while True:
        k += 1
        current_state = game.reset()

        game_states = [[], []]
        game_actions = [[], []]

        while not game.over:
            for i in range(2):
                if np.random.random() > .90 and len(history) >= MIN_HISTORY:
                    T = torch.from_numpy(current_state.reshape((1, 6, BOARD_SIZE, BOARD_SIZE))).float().to(device)
                    pred = net(T)[0].cpu().detach().numpy()
                    action = pred_to_action(pred)

                    if not game.action_is_valid(game.players[i], action):
                        action = game.get_random_action(i)

                else:
                    action = game.get_random_action(i)

                current_state, done = game.step(i, action)
                action = action_to_pred(action)
                game_states[i].append(current_state)
                game_actions[i].append(action)

                if done:
                    for state, action in zip(game_states[i], game_actions[i]):
                        history.append((state, action))
                    break

        if len(history) >= MIN_HISTORY:
            samples = random.sample(history, BATCH_SIZE)
            states = torch.from_numpy(np.array([sample[0] for sample in samples])).float().to(device)
            actions = torch.from_numpy(np.array([sample[1] for sample in samples])).float().to(device)

            optimizer.zero_grad()

            outputs = net(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            print(f'Training loss: {sum(losses) / len(losses):.5f} | Memory size: {len(history)} | Games: {k}')

        if k % 180 == 99:
            torch.save(net.state_dict(), 'net.pth')
