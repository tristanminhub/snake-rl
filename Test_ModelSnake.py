import random
import torch
import torch.nn as nn
from snake_game_base import SnakeGame
from snake_viewer import SnakeViewer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 295
        output_size = 4
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
            )
    def forward(self, x):
        x = self.model(x)
        return x

MIN_SIZE = 10000
batch_size = 64
GRID_WIDTH = 25
GRID_HEIGHT = 20

class ReplayBuffer:
    def __init__(self):
        self.capacity = 100000
        self.buffer = []
        self.index = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

replay_buffer = ReplayBuffer()
q_network = DQN().to(device)
target_q_network = DQN().to(device)
target_q_network.load_state_dict(q_network.state_dict())


viewer = SnakeViewer()
optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
gamma = 0.98
time_view = 500
nb_envs = 512
directions = ["Up", "Down", "Left", "Right"]
total_step = 100000
rayon = 8

class Batch():
    def __init__(self, nb_envs):
        self.games = SnakeGame(nb_envs, device)
        
    def play_step(self, temp):
        score_before = self.games.score.clone()
 
        states = get_state(self.games)  # (nb_envs, 295)

        q_values = q_network(states)
        probs = torch.softmax(q_values / temp, dim=1)
        actions = torch.multinomial(probs, num_samples=1).squeeze()  # (nb_envs,)

        self.games.set_direction(actions) 
        self.games.move_snake()

        score_after = self.games.score
        done = self.games.game_over
 
        reward = torch.full((nb_envs,), -0.1, dtype=torch.float32, device=device)
        reward[score_after > score_before] = 10.0
        reward[done] = -15.0
 
        next_states = get_state(self.games)  # (nb_envs, 295)
 
        distance_before = abs(states[:, 0]) + abs(states[:, 1])
        distance_after  = abs(next_states[:, 0]) + abs(next_states[:, 1])
 
        closer  = ~done & (distance_after < distance_before)
        farther = ~done & (distance_after > distance_before)
        reward[closer]  += 0.1 * distance_weight
        reward[farther] -= 0.1 * distance_weight
 
        return states, actions, reward, next_states, done

    def train(self):
        with torch.no_grad():
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.cat(list(states), dim=0)
            next_states = torch.cat(list(next_states), dim=0)
            rewards = torch.cat(list(rewards), dim=0)
            actions = torch.cat(list(actions), dim=0)
            dones = torch.cat(list(dones), dim=0)
            next_all_q = target_q_network(next_states)
            y_target = rewards + gamma * torch.max(next_all_q,dim=1).values
            y_target[dones] = rewards[dones]    
            
        all_q_values = q_network(states)
        y_pred = all_q_values[torch.arange(batch_size*nb_envs), actions]

        self.loss = loss_fn(y_pred, y_target)

        optimizer.zero_grad()
        self.loss.backward()
        optimizer.step()

        if step % time_view == 0:
            target_q_network.load_state_dict(q_network.state_dict())

def get_state(games):
    head_x = games.bodies[:, 0, 0].float()
    head_y = games.bodies[:, 0, 1].float()
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
    temp = max(0.1, 1 - step / total_step)
    distance_weight = max(0, 1 - step / 20000)
    states, actions, reward, next_states, done = batch.play_step(temp)
    replay_buffer.push(states, actions, reward, next_states, done)

    if batch.games.game_over.any():
        finished_score_total += batch.games.score[done].sum().item()
        finished_score_count += done.sum().item()
        best_finished_score = max(best_finished_score, batch.games.score[done].max().item())
        batch.games.reset_env(done)
    
    viewer.draw(batch.games.get_snake_list(0), batch.games.get_food(0))

    if len(replay_buffer.buffer) >= MIN_SIZE:
        batch.train()
        loss_total += batch.loss.item() 
        loss_count += 1

        score_mean = finished_score_total / finished_score_count if finished_score_count > 0 else 0
        loss_mean = loss_total / loss_count if loss_count > 0 else 0

        if step % time_view ==0:
            score_mean = finished_score_total / finished_score_count if finished_score_count > 0 else 0
            loss_mean = loss_total / loss_count if loss_count > 0 else 0
            print(f"Step : {step}, Score Moyen : {score_mean:.2f}, Loss Moyenne : {loss_mean:.2f}, Best Score : {best_finished_score}")
            loss_total = 0
            best_finished_score = 0
            finished_score_total = 0
            finished_score_count = 0
            loss_count = 0
        
        
