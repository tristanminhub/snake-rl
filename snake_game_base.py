import torch
import random

GRID_WIDTH = 25
GRID_HEIGHT = 20
MAX_SNAKE_LENGTH = GRID_WIDTH * GRID_HEIGHT

class SnakeGame:

    def __init__(self, nb_envs=512, device=None):
        self.nb_envs = nb_envs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._reset_all()

    def _reset_all(self):
        n, W, H, D = self.nb_envs, GRID_WIDTH, GRID_HEIGHT, self.device

        self.bodies = torch.zeros(n, MAX_SNAKE_LENGTH, 2, dtype=torch.long, device=D)
        self.bodies[:, 0, 0] = W // 2
        self.bodies[:, 0, 1] = H // 2

        self.lengths = torch.ones(n, dtype=torch.long, device=D)

        self.directions = torch.full((n,), 3, dtype=torch.long, device=D)

        self.score = torch.zeros(n, dtype=torch.long, device=D)

        self.game_over = torch.zeros(n, dtype=torch.bool, device=D)

        self._grid = torch.zeros(n, W, H, dtype=torch.bool, device=D)
        self._grid[:, W // 2, H // 2] = True

        self.food = torch.zeros(n, 2, dtype=torch.long, device=D)
        self._place_food(torch.ones(n, dtype=torch.bool, device=D))

    def _place_food(self, mask):
        indices = mask.nonzero(as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            return
        for i in indices.tolist():
            self._place_food_single(i)

    def _place_food_single(self, i):
        W, H = GRID_WIDTH, GRID_HEIGHT
        occupied = self._grid[i] 
        free = (~occupied).nonzero(as_tuple=False) 
        if free.shape[0] == 0:
            self.game_over[i] = True
            return
        idx = random.randrange(free.shape[0])
        self.food[i, 0] = free[idx, 0]
        self.food[i, 1] = free[idx, 1]

    def set_direction(self, new_directions):

        opposites = torch.tensor([1, 0, 3, 2], device=self.device)
        opp = opposites[self.directions]
        valid = new_directions != opp
        self.directions = torch.where(valid, new_directions, self.directions)

    def move_snake(self):
        n, W, H, D = self.nb_envs, GRID_WIDTH, GRID_HEIGHT, self.device
        alive = ~self.game_over

        dx = torch.tensor([0, 0, -1, 1], device=D)
        dy = torch.tensor([-1, 1, 0, 0], device=D)
        head = self.bodies[:, 0, :]  # (n, 2)
        new_hx = head[:, 0] + dx[self.directions]
        new_hy = head[:, 1] + dy[self.directions]

        wall = (new_hx < 0) | (new_hx >= W) | (new_hy < 0) | (new_hy >= H)

        safe_x = new_hx.clamp(0, W - 1)
        safe_y = new_hy.clamp(0, H - 1)
        body_hit = self._grid[torch.arange(n, device=D), safe_x, safe_y]

        dead = alive & (wall | body_hit)
        self.game_over |= dead

        ate = alive & ~dead & (new_hx == self.food[:, 0]) & (new_hy == self.food[:, 1])
        self.score += ate.long()

        moving = alive & ~dead

        no_eat_moving = moving & ~ate
        if no_eat_moving.any():
            idx = no_eat_moving.nonzero(as_tuple=False).squeeze(1)
            tail_pos = self.lengths[idx] - 1
            tx = self.bodies[idx, tail_pos, 0]
            ty = self.bodies[idx, tail_pos, 1]
            self._grid[idx, tx, ty] = False
            self.lengths[idx] 

        if moving.any():
            idx = moving.nonzero(as_tuple=False).squeeze(1)
            max_len = self.lengths[idx].max().item()
            self.bodies[idx, 1:max_len + 1] = self.bodies[idx, 0:max_len].clone()
            self.bodies[idx, 0, 0] = new_hx[idx]
            self.bodies[idx, 0, 1] = new_hy[idx]
            self._grid[idx, new_hx[idx], new_hy[idx]] = True

        if ate.any():
            idx = ate.nonzero(as_tuple=False).squeeze(1)
            self.lengths[idx] += 1
            self._place_food(ate)

    def is_collision(self, positions):
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        px, py = positions[:, 0], positions[:, 1]
        wall = (px < 0) | (px >= GRID_WIDTH) | (py < 0) | (py >= GRID_HEIGHT)
        safe_x = px.clamp(0, GRID_WIDTH - 1)
        safe_y = py.clamp(0, GRID_HEIGHT - 1)
        body = self._grid[torch.arange(self.nb_envs, device=self.device), safe_x, safe_y]
        return wall | body

    def get_vision(self, rayon):
        head_x = self.bodies[:, 0, 0]
        head_y = self.bodies[:, 0, 1]

        offsets = torch.arange(-rayon, rayon + 1, device=self.device)
        px = head_x.view(-1, 1, 1) + offsets.view(1, 1, -1)
        py = head_y.view(-1, 1, 1) + offsets.view(1, -1, 1)

        wall = (px < 0) | (px >= GRID_WIDTH) | (py < 0) | (py >= GRID_HEIGHT)
        safe_x = px.clamp(0, GRID_WIDTH - 1)
        safe_y = py.clamp(0, GRID_HEIGHT - 1)
        body = self._grid[
            torch.arange(self.nb_envs, device=self.device).view(-1, 1, 1).expand_as(safe_x),
            safe_x, safe_y
        ]
        return (wall | body).float().view(self.nb_envs, -1)
    
    def reset_env(self, mask):
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        W, H, D = GRID_WIDTH, GRID_HEIGHT, self.device
        self._grid[idx] = False
        self.bodies[idx] = 0
        self.bodies[idx, 0, 0] = W // 2
        self.bodies[idx, 0, 1] = H // 2
        self.lengths[idx] = 1
        self.directions[idx] = 3
        self.score[idx] = 0
        self.game_over[idx] = False
        self._grid[idx, W // 2, H // 2] = True
        self._place_food(mask)

    def get_snake_list(self, i):
        length = self.lengths[i].item()
        return [(self.bodies[i, j, 0].item(), self.bodies[i, j, 1].item())
                for j in range(length)]

    def get_food(self, i):
        return (self.food[i, 0].item(), self.food[i, 1].item())