import torch
import torch.nn as nn
from snake_game_base import SnakeGame
from snake_viewer import SnakeViewer
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SnakeNN(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 10
        output_size = 4
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64,output_size),
            )
    def forward(self, x):
        x = self.model(x)
        return x

model = SnakeNN().to(device)
viewer = SnakeViewer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
gamma = 0.9
time_view = 50
nb_envs = 256
directions = ["Up", "Down", "Left", "Right"]


class Batch():
    def __init__(self, nb_envs):
        self.games = [SnakeGame() for _ in range(nb_envs)]
        
    def play_step(self):
        self.reward = torch.zeros(nb_envs, dtype=torch.float32, device=device)
        self.done = torch.zeros(nb_envs, dtype=torch.bool, device=device)
        states = torch.stack([get_state(game) for game in self.games])
        self.output = model(states)
        probs = torch.softmax(self.output, dim=1)
        self.actions = torch.multinomial(probs, 1).squeeze()
        direction = [directions[action.item()] for action in self.actions]

        distance_head_food1 = abs(states[:, 0]) + abs(states[:, 1])

        for i, game in enumerate(self.games):

            score_before = game.score
            
            game.set_direction(direction[i])
            game.move_snake()

            score_after = game.score

            if game.game_over:
                self.reward[i] = -10
                self.done[i] = True
            elif score_after > score_before :
                self.reward[i] +=1
            else : 
                self.reward[i] -= 0.1

        self.next_states = torch.stack([get_state(game) for game in self.games])
        self.next_output = model(self.next_states)

        distance_head_food2 = abs(self.next_states[:, 0]) + abs(self.next_states[:, 1])

        for i in range(nb_envs):
            if not self.done[i]:
                if distance_head_food2[i] < distance_head_food1[i]:
                    self.reward[i] += 0.1
                elif distance_head_food2[i] > distance_head_food1[i]:
                    self.reward[i] -= 0.1

        with torch.no_grad():
            self.next_q = gamma * self.next_output.max(dim=1).values

    def train(self):
        target = self.output.clone().detach()

        target_q = self.reward + self.next_q
        target_q[self.done] = self.reward[self.done]

        target[torch.arange(nb_envs, device=device), self.actions] = target_q

        self.loss = loss_fn(self.output, target)

        optimizer.zero_grad()
        self.loss.backward()
        optimizer.step()

def get_state(game):
    food_x, food_y = game.food
    head_x, head_y = game.snake[0]
    food_dx = food_x - head_x
    food_dy = food_y - head_y
    danger_right = game.is_collision((head_x +1, head_y))
    danger_left = game.is_collision((head_x -1, head_y))
    danger_down = game.is_collision((head_x, head_y +1))
    danger_up = game.is_collision((head_x, head_y -1))
    dir_up = game.direction == "Up"
    dir_down = game.direction == "Down"
    dir_left = game.direction == "Left"
    dir_right = game.direction == "Right"

    x =torch.tensor([
        food_dx,
        food_dy,
        danger_up,
        danger_down,
        danger_left,
        danger_right,
        dir_up,
        dir_down,
        dir_left,
        dir_right],
        dtype=torch.float32, device=device)
    
    return x

batch = Batch(nb_envs)
loss_total = 0
loss_count = 0
score_total =0

for episode in range(100000):
    batch.play_step()
    batch.train()
    for i, game in enumerate(batch.games):
        if game.game_over:
            batch.games[i] = SnakeGame()

    loss_total += batch.loss.item() 
    loss_count += 1

    score_total += sum(game.score for game in batch.games) / nb_envs

    if episode % time_view ==0:
        score_mean = score_total / loss_count
        loss_mean = loss_total / loss_count if loss_count > 0 else 0
        print(f"Episode : {episode}, Score : {score_mean:.2f}, Loss : {loss_mean:.2f}")
        loss_total = 0
        loss_mean = 0
        score_total = 0
        loss_count = 0

    viewer.draw(batch.games[0])
    time.sleep(0.02)



