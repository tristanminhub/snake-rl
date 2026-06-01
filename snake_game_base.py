import random

GRID_WIDTH = 25
GRID_HEIGHT = 20


class SnakeGame:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = "Right"
        self.score = 0
        self.game_over = False
        self.place_food()

    def place_food(self):
        available_cells = [
            (x, y)
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
            if (x, y) not in self.snake
        ]

        if available_cells:
            self.food = random.choice(available_cells)
        else:
            self.game_over = True

    def move_snake(self):
        if self.game_over:
            return

        new_head = self.get_next_head_position()

        if self.is_collision(new_head):
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

    def get_next_head_position(self):
        head_x, head_y = self.snake[0]

        if self.direction == "Up":
            head_y -= 1
        elif self.direction == "Down":
            head_y += 1
        elif self.direction == "Left":
            head_x -= 1
        elif self.direction == "Right":
            head_x += 1

        return head_x, head_y

    def is_collision(self, position):
        x, y = position

        if x < 0 or x >= GRID_WIDTH:
            return True

        if y < 0 or y >= GRID_HEIGHT:
            return True

        if position in self.snake:
            return True

        return False
    
    def set_direction(self, new_direction):
        opposites = {
            "Up": "Down",
            "Down": "Up",
            "Left": "Right",
            "Right": "Left",
        }

        if new_direction != opposites[self.direction]:
            self.direction = new_direction