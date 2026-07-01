import random
import torch
import torch.nn as nn
from snake_game_base import SnakeGame
from snake_viewer import SnakeViewer
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 295
        output_size = 4
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    def forward(self, x):
        return self.model(x)

MIN_SIZE = 50000
batch_size = 1024
GRID_WIDTH = 25
GRID_HEIGHT = 25
capacity = 5120000*2
n_steps = 20

class ReplayBuffer:
    def __init__(self):
        self.capacity = capacity
        self.states = torch.zeros((capacity, 295))
        self.next_states = torch.zeros((capacity, 295))
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.rewards = torch.zeros((capacity, 1))
        self.dones = torch.zeros((capacity, 1))
        self.index = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        i = self.index
        n = state.shape[0]

        self.states[i:i+n] = state.cpu()
        self.next_states[i:i+n] = next_state.cpu()
        self.actions[i:i+n] = action.unsqueeze(-1).cpu()
        self.rewards[i:i+n] = reward.unsqueeze(-1).cpu()
        self.dones[i:i+n] = done.float().unsqueeze(-1).cpu()

        self.index = (i + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.states[idx].to(device),
            self.actions[idx].to(device),
            self.rewards[idx].to(device),
            self.next_states[idx].to(device),
            self.dones[idx].to(device),
        )

replay_buffer = ReplayBuffer()
q_network = DQN().to(device)
target_q_network = DQN().to(device)
target_q_network.load_state_dict(q_network.state_dict())

q_network.load_state_dict(torch.load("snake_dqn_1M.pth"))
target_q_network.load_state_dict(torch.load("snake_dqn_1M.pth"))

viewer = SnakeViewer()
optimizer = torch.optim.Adam(q_network.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()
gamma = 0.99
time_view = 500
nb_envs = 512
directions = ["Up", "Down", "Left", "Right"]
total_step = 400000
rayon = 8

class Batch():
    def __init__(self, nb_envs):
        self.games = SnakeGame(nb_envs, device)

    def play_step(self, temp):
        score_before = self.games.score.clone()

        states = get_state(self.games)

        q_values = q_network(states)
        probs = torch.softmax(q_values / temp, dim=1)
        actions = q_values.argmax(dim=1)

        self.games.set_direction(actions)
        self.games.move_snake()

        score_after = self.games.score
        done = self.games.game_over.clone()

        # # à l'init du training loop
        # mobility_history = torch.zeros(nb_envs, 5, dtype=torch.long, device=device)

        # current_mobility = self.games.compute_mobility()
        # mobility_drop = mobility_history[:, 0] - current_mobility
        # penalty_mask = mobility_drop > 4
        
        # mobility_history = torch.roll(mobility_history, shifts=-1, dims=1)
        # mobility_history[:, -1] = current_mobility


        reward = torch.full((nb_envs,), -0.1, dtype=torch.float32, device=device)
        reward[score_after > score_before] = 15.0
        reward[done] = -75.0
        # reward[penalty_mask] -= 5.0

        return states, actions, reward, done

    def train(self):
        with torch.no_grad():
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            best_actions = q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = target_q_network(next_states)
            best_next_q_values = next_q_values.gather(1, best_actions).squeeze(1)
            y_target = rewards.squeeze() + (gamma ** n_steps) * best_next_q_values * (1 - dones.squeeze())
        q_values = q_network(states)
        best_q_value = torch.gather(q_values, 1, actions).squeeze()
        self.loss = loss_fn(best_q_value, y_target)
        optimizer.zero_grad()
        self.loss.backward()
        optimizer.step()
        if step % 2500 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

def get_state(games):
    food_dx = (games.food[:, 0] - games.bodies[:, 0, 0]).float()
    food_dy = (games.food[:, 1] - games.bodies[:, 0, 1]).float()

    dir_onehot = torch.zeros(nb_envs, 4, dtype=torch.float32, device=device)
    dir_onehot.scatter_(1, games.directions.unsqueeze(1), 1.0)

    vision = games.get_vision(rayon)

    return torch.cat([
        food_dx.unsqueeze(1),
        food_dy.unsqueeze(1),
        dir_onehot,
        vision
    ], dim=1)

batch = Batch(nb_envs)
loss_total = 0
loss_count = 0
finished_score_total = 0
finished_score_count = 0
best_finished_score = 0

for step in range(total_step):
    temp = max(0.05, (1 - step / total_step))

    initial_states = get_state(batch.games)
    first_actions = None
    n_step_rewards = torch.zeros(nb_envs, device=device)
    n_step_dones = torch.zeros(nb_envs, dtype=torch.bool, device=device)

    for i in range(n_steps):
        states, actions, reward, done = batch.play_step(temp)

        if i == 0:
            first_actions = actions

        mask = ~n_step_dones
        n_step_rewards[mask] += (gamma ** i) * reward[mask]

        if done.any():
            new_deaths = done & ~n_step_dones
            if new_deaths.any():
                scores_done = batch.games.score[new_deaths].clone()
                finished_score_total += scores_done.sum().item()
                finished_score_count += new_deaths.sum().item()
                best_finished_score = max(best_finished_score, scores_done.max().item())
            batch.games.reset_env(done)

        n_step_dones |= done

    final_next_states = get_state(batch.games)

    replay_buffer.push(initial_states, first_actions, n_step_rewards, final_next_states, n_step_dones)

    viewer.draw(batch.games.get_snake_list(0), batch.games.get_food(0))

    if replay_buffer.size >= MIN_SIZE:
        batch.train()
        loss_total += batch.loss.item()
        loss_count += 1

        if step % time_view == 0:
            score_mean = finished_score_total / finished_score_count if finished_score_count > 0 else 0
            loss_mean = loss_total / loss_count if loss_count > 0 else 0
            print(f"Step : {step}, Score Moyen : {score_mean:.2f}, Loss Moyenne : {loss_mean:.2f}, Best Score : {best_finished_score}")
            loss_total = 0
            best_finished_score = 0
            finished_score_total = 0
            finished_score_count = 0
            loss_count = 0


    if step % 50000 == 0:
        torch.save(q_network.state_dict(), "snake_dqn_1M.pth")
