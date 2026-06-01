import tkinter as tk
from snake_game_base import GRID_WIDTH, GRID_HEIGHT

CELL_SIZE = 20


class SnakeViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Snake RL Viewer")

        self.canvas = tk.Canvas(
            self.root,
            width=GRID_WIDTH * CELL_SIZE,
            height=GRID_HEIGHT * CELL_SIZE,
            bg="black"
        )
        self.canvas.pack()

    def draw(self, game):
        self.canvas.delete("all")

        # pomme
        food_x, food_y = game.food
        self.draw_cell(food_x, food_y, "red")

        # serpent
        for i, (x, y) in enumerate(game.snake):
            color = "lime" if i == 0 else "green"
            self.draw_cell(x, y, color)

        # score
        self.canvas.create_text(
            10,
            10,
            anchor="nw",
            text=f"Score: {game.score}",
            fill="white",
            font=("Arial", 14, "bold")
        )

        if game.game_over:
            self.canvas.create_text(
                GRID_WIDTH * CELL_SIZE / 2,
                GRID_HEIGHT * CELL_SIZE / 2,
                text="Game Over",
                fill="white",
                font=("Arial", 28, "bold")
            )

        self.root.update()

    def draw_cell(self, x, y, color):
        self.canvas.create_rectangle(
            x * CELL_SIZE,
            y * CELL_SIZE,
            (x + 1) * CELL_SIZE,
            (y + 1) * CELL_SIZE,
            fill=color,
            outline="#111"
        )