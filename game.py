import random
from collections import deque
from copy import deepcopy

import numpy as np
import pygame
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, MSELoss
from torch.optim import Adam

from astar import astar

BOARD_SIZE = 9
DISPLAY_SIZE = 640


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

        self.display = None
        self.clock = None
        self.render_init = False
        self.last_frame = None

        self.reset()

    def reset(self):
        self.over = False

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                self.cells[y][x] = Cell((x, y))

        self.players[:] = (TopPlayer((4, 0), 0), BottomPlayer((4, 8), 1))
        for player in self.players:
            self.cells[player.y][player.x].player = player

    def render(self, slow=False):
        if not self.render_init:
            pygame.init()
            pygame.display.set_caption('Quoridor-AI')
            self.display = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
            self.clock = pygame.time.Clock()
            self.display.fill((200, 200, 200))
            self.last_frame = pygame.time.get_ticks()
            self.render_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()

        now = pygame.time.get_ticks()
        if now - self.last_frame < 1000 / 90:
            return

        self.display.fill((200, 200, 200))

        rect_size = (DISPLAY_SIZE / BOARD_SIZE)
        player_size = rect_size // 1.2
        padding = (rect_size - player_size) // 2

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                cell = self.cells[y][x]

                if cell.player is not None:
                    color = (120, 0, 0) if cell.player.no == 0 else (0, 0, 120)
                    pygame.draw.rect(self.display, color, pygame.Rect(x * rect_size + padding,
                                                                      y * rect_size + padding,
                                                                      player_size,
                                                                      player_size))
                if cell.f_left:
                    pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(x * rect_size,
                                                                          y * rect_size,
                                                                          padding,
                                                                          rect_size))
                if cell.f_up:
                    pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(x * rect_size,
                                                                          y * rect_size,
                                                                          rect_size,
                                                                          padding))
                if cell.f_right:
                    pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect((x + 1) * rect_size - padding,
                                                                          y * rect_size,
                                                                          padding,
                                                                          rect_size))
                if cell.f_down:
                    pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(x * rect_size,
                                                                          (y + 1) * rect_size - padding,
                                                                          rect_size,
                                                                          padding))

        self.last_frame = now
        pygame.display.update()
        if slow:
            self.clock.tick(30)

    def get_random_action(self, player_id):
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

            valid = self.action_is_valid(player_id, action)
        return action

    def action_is_valid(self, player_id, action):
        player = self.players[player_id]

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
                     not self.cells[origin_y + 1][origin_x].f_left and
                     not (self.cells[origin_y][origin_x].f_down and self.cells[origin_y][origin_x - 1].f_down))
                    or (not vertical and origin_y > 0 and origin_x + 1 < BOARD_SIZE and
                        not self.cells[origin_y][origin_x].f_up and
                        not self.cells[origin_y][origin_x + 1].f_up and
                        not (self.cells[origin_y][origin_x].f_right and self.cells[origin_y - 1][origin_x].f_right))
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

    def place_fence(self, player_id, action):
        player = self.players[player_id]
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
    def move(self, player_id, action):
        player = self.players[player_id]
        player_cell = self.cells[player.y][player.x]
        _, direction = action

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
            self.move(player_id, action)
        else:
            self.place_fence(player_id, action)

        if player.won():
            self.over = True

        return self.over

    @staticmethod
    def mirror_state(state):
        mirrored = state[:, ::-1, ::-1]
        return mirrored.copy()

    @staticmethod
    def mirror_action(action):
        mirrored = action.copy()
        if action[0] == 0:
            mirrored[1] = (action[1] + 2) % 4
        else:
            for i in range(1, 3):
                mirrored[i] = (BOARD_SIZE - 1) - action[i]
            mirrored[2 - action[3]] -= 1
        return mirrored

    def get_state(self, player_id):
        player = self.players[player_id]
        state = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.bool)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                cell = self.cells[y][x]
                state[0][y][x] = cell.f_left
                state[1][y][x] = cell.f_up
                state[2][y][x] = cell.f_right
                state[3][y][x] = cell.f_down
                state[4][y][x] = cell.player is not None
                state[5][y][x] = cell.player is not None and cell.player == player
        return self.mirror_state(state) if player_id else state


def action_to_pred(action):
    pred = np.zeros(22, dtype=np.bool)
    pred[action[0]] = 1

    if pred[0]:
        pred[2 + action[1]] = 1
    else:
        pred[2 + action[1]] = 1
        pred[11 + action[2]] = 1
        pred[20 + action[3]] = 1
    return pred


def pred_to_action(pred):
    action = [np.argmax(pred[:2])]
    if action[0] == 0:
        action.append(np.argmax(pred[2: 6]))
    else:
        action.append(np.argmax(pred[2:11]))
        action.append(np.argmax(pred[11:20]))
        action.append(np.argmax(pred[20:22]))
    return action


"""
    output: (21,)
    [0, left, up right, down, ...]
    [1, x=0...x=8, y=0...y=8, h, v]
"""


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = Linear(6 * 9 * 9, 1024)
        self.linear2 = Linear(1024, 1024)
        self.linear3 = Linear(1024, 22)

    def forward(self, x):
        x = x.view(-1, 6 * 9 * 9)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    game = Quoridor()
    RENDER = True
    SLOW = False
    RENDER_SLOW_EVERY = 32

    MIN_HISTORY = 4096
    MAX_HISTORY = 16_384
    history = deque(maxlen=MAX_HISTORY)

    LOSS_HISTORY = 128
    losses = deque([1], maxlen=LOSS_HISTORY)

    VALID_ACTION_HISTORY = 1_024
    valid_actions = deque([False], maxlen=VALID_ACTION_HISTORY)

    REWARD_HISTORY = 1_024
    rewards = deque([0], maxlen=REWARD_HISTORY)

    epsilon = .99
    EPSILON_DECAY = .9999
    MIN_EPSILON = .005

    BATCH_SIZE = 32
    LR = 0.001
    DISCOUNT = .80

    WON_REWARD = 1.0
    LOST_REWARD = -1.0
    STEP_REWARD = -.01
    ERROR_REWARD = -2.0

    LOAD_NET = False
    SAVE_INTERVAL = 512
    TARGET_INTERVAL = 1_024

    net = Net().to(device)
    pp = 0
    for p in list(net.parameters()):
        nn = 1
        for s in list(p.size()):
            nn *= s
        pp += nn
    print(f'Number of parameters: {pp}')
    target_net = Net().to(device)
    if LOAD_NET:
        print('Loading prev net')
        net.load_state_dict(torch.load('net.pth'))
    target_net.load_state_dict(net.state_dict())

    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=LR)

    k = 0
    while True:
        game.reset()

        last_current_states = [None, None]
        last_preds = [None, None]
        last_valid = [None, None]
        game_history = []

        turns = 0
        while not game.over and turns < 512:
            k += 1
            turns += 1

            for i in range(2):
                state = game.get_state(i)
                current_state = torch.from_numpy(state).float()

                valid_action = True
                if (LOAD_NET or len(history) >= MIN_HISTORY) and np.random.random() > epsilon:
                    with torch.no_grad():
                        T = current_state.clone().unsqueeze(0).to(device)
                        preds = net(T)[0].cpu()
                        action = pred_to_action(preds.numpy())
                        action = game.mirror_action(action) if i else action

                    valid_action = game.action_is_valid(i, action)
                    valid_actions.append(valid_action)

                else:
                    action = game.get_random_action(i)
                    preds = torch.tensor(action_to_pred(game.mirror_action(action) if i else action),
                                         dtype=torch.float)

                done = game.step(i, action) if valid_action else False

                if done:
                    T_reward = torch.tensor(WON_REWARD, dtype=torch.float)
                    T_done = torch.tensor(done, dtype=torch.float)

                    history.append((current_state.clone(), last_preds[i].clone(), T_reward.clone(),
                                    current_state.clone(), T_done.clone()))
                    rewards.append(WON_REWARD)

                    T_current_state_l = last_current_states[abs(i - 1)]
                    T_reward_l = torch.tensor(LOST_REWARD, dtype=torch.float)
                    T_done_l = torch.tensor(done, dtype=torch.float)

                    history.append((T_current_state_l.clone(), last_preds[abs(i - 1)].clone(), T_reward_l.clone(),
                                    T_current_state_l.clone(), T_done_l.clone()))

                elif last_current_states[i] is not None:
                    r = STEP_REWARD if last_valid[i] else ERROR_REWARD
                    T_reward = torch.tensor(r, dtype=torch.float)
                    T_done = torch.tensor(done, dtype=torch.float)

                    history.append((last_current_states[i].clone(), last_preds[i].clone(), T_reward.clone(),
                                    current_state.clone(), T_done.clone()))
                    rewards.append(r)

                last_current_states[i] = current_state.clone()
                last_preds[i] = preds.clone()
                last_valid[i] = valid_action

                if len(history) >= MIN_HISTORY:
                    if RENDER:
                        game.render(slow=SLOW and k % RENDER_SLOW_EVERY == 0)

                    if epsilon > MIN_EPSILON:
                        epsilon *= EPSILON_DECAY
                        epsilon = max(MIN_EPSILON, epsilon)

                    minibatch = random.sample(history, BATCH_SIZE)
                    T_current_states, T_actions, T_rewards, T_new_current_states, T_dones = zip(*minibatch)

                    T_current_states = torch.stack(T_current_states).to(device)
                    T_actions = torch.stack(T_actions).to(device)
                    T_rewards = torch.stack(T_rewards).to(device)
                    T_new_current_states = torch.stack(T_new_current_states).to(device)
                    T_dones = torch.stack(T_dones).to(device)

                    with torch.no_grad():
                        T_current_qs_list = net(T_current_states)
                        T_future_qs_list = target_net(T_new_current_states)

                    X = torch.empty((BATCH_SIZE, 6, BOARD_SIZE, BOARD_SIZE), dtype=torch.float, device=device)
                    y = torch.empty((BATCH_SIZE, 22), dtype=torch.float, device=device)

                    for index in range(BATCH_SIZE):
                        T_current_state = T_current_states[index]
                        T_action = T_actions[index]
                        T_reward = T_rewards[index]
                        T_new_current_state = T_new_current_states[index]
                        T_done = T_dones[index]

                        T_current_qs = T_current_qs_list[index]
                        T_future_qs = T_future_qs_list[index]

                        if action[0] == 0:
                            indexes = [(0, 2), (2, 5)]
                        else:
                            indexes = [(0, 2), (2, 9), (11, 9), (20, 2)]

                        for start, length in indexes:
                            if not done:
                                T_max_future_q = torch.max(T_future_qs[start: start + length])
                                new_q = T_reward + DISCOUNT * T_max_future_q
                            else:
                                new_q = T_reward

                            T_current_qs[start + torch.argmax(T_action[start: start + length])] = new_q

                        X[index] = T_current_state
                        y[index] = T_current_qs

                    optimizer.zero_grad()

                    outputs = net(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

                if done:
                    break

            if len(history) >= MIN_HISTORY:
                if k % TARGET_INTERVAL == TARGET_INTERVAL - 1:
                    target_net.load_state_dict(net.state_dict())

                if k % SAVE_INTERVAL == SAVE_INTERVAL - 1:
                    torch.save(net.state_dict(), 'net.pth')

        if len(history) >= MIN_HISTORY:
            print(f'Training loss: {sum(losses) / len(losses):.5f} | '
                  f'Epsilon: {epsilon:.5f} | '
                  f'Mean reward: {sum(rewards) / len(rewards):.5f} | '
                  f'Valid actions: {(sum(valid_actions) / len(valid_actions)) * 100:.1f}% | '
                  f'Memory size: {len(history)} | Steps: {k}')
