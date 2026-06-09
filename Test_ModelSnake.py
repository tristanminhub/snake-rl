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
        input_size = 127
        output_size = 4
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
            )
    def forward(self, x):
        x = self.model(x)
        return x

model = SnakeNN().to(device)
viewer = SnakeViewer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
gamma = 0.95
time_view = 50
nb_envs = 512
directions = ["Up", "Down", "Left", "Right"]
total_step = 40000
rayon = 5



class Batch():
    def __init__(self, nb_envs):
        self.games = [SnakeGame() for _ in range(nb_envs)]
        
    def play_step(self, temp):
        self.reward = torch.zeros(nb_envs, dtype=torch.float32, device=device)
        self.done = torch.zeros(nb_envs, dtype=torch.bool, device=device)
        states = torch.stack([get_state(game) for game in self.games])
        self.output = model(states)
        probs = torch.softmax(self.output/temp, dim=1)
        self.actions = torch.multinomial(probs, num_samples=1).squeeze()
        direction = [directions[action.item()] for action in self.actions]

        distance_head_food1 = abs(states[:, 0]) + abs(states[:, 1])

        for i, game in enumerate(self.games):

            score_before = game.score
            
            game.set_direction(direction[i])
            game.move_snake()

            score_after = game.score

            if game.game_over:
                self.reward[i] = -30
                self.done[i] = True
            elif score_after > score_before :
                self.reward[i] +=10
            else : 
                self.reward[i] -= 0.1

        self.next_states = torch.stack([get_state(game) for game in self.games])

        distance_head_food2 = abs(self.next_states[:, 0]) + abs(self.next_states[:, 1])

        for i in range(nb_envs):
             if not self.done[i]:
                 if distance_head_food2[i] < distance_head_food1[i]:
                     self.reward[i] += 0.1 * distance_weight
                 elif distance_head_food2[i] > distance_head_food1[i]:
                     self.reward[i] -= 0.1 * distance_weight

        with torch.no_grad():
            self.next_output = model(self.next_states)
            self.next_q = gamma * torch.max(self.next_output, dim=1).values

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
    vision = []
    dir_up = game.direction == "Up"
    dir_down = game.direction == "Down"
    dir_left = game.direction == "Left"
    dir_right = game.direction == "Right"
    

    for y in range(head_y - rayon, head_y + rayon +1):
        for x in range(head_x - rayon, head_x + rayon +1):
            vision.append((game.is_collision((x, y))))
    food_dx = food_x - head_x
    food_dy = food_y - head_y

    state = [
        food_dx,
        food_dy,
        dir_up,
        dir_down,
        dir_left,
        dir_right
    ]
    
    for i in vision:
        state.append(i)

    x = torch.tensor(state, dtype=torch.float32, device=device)

    return x

batch = Batch(nb_envs)
loss_total = 0
loss_count = 0
finished_score_total = 0
finished_score_count = 0
best_finished_score = 0


for step in range(total_step):
    temp = max(0.1, 1 - 2*step / total_step)
    distance_weight = max(0, 1 - step / 7000)
    batch.play_step(temp)
    batch.train()
    for i, game in enumerate(batch.games):
        if game.game_over:
            finished_score_total += game.score
            finished_score_count += 1
            best_finished_score = max(best_finished_score, game.score)
            batch.games[i] = SnakeGame()

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

    viewer.draw(batch.games[0])
