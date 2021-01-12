"""A command line version of Minesweeper"""

from copy import deepcopy
import numpy
import random
import re
import time
from itertools import product
from typing import NamedTuple


class Coords(NamedTuple):
    row: int
    col: int

    def adjacent(self, other):
        """Returns true if both cells are equal or adjacent"""
        return self != other and (
            abs(self.row - other.row) <= 1 and abs(self.col - other.col) <= 1
        )


class Game:

    props_per_cell = 2

    MINE = "M"

    VISIBLE_INDEX = 0
    ADJACENT_INDEX = 1

    def __init__(
        self, width: int, height: int, number_of_mines: int, initial_moves: int = 0
    ):

        # basic sanity check, reasonable games should never be close to this
        assert number_of_mines <= width * height - 9

        self.width = width
        self.height = height
        self.number_of_mines = number_of_mines

        self.cells_hidden = width * height
        self.number_of_flags = 0
        self.game_over = False

        # The point is to store the player's grid as a 3-dim'l numpy array so that it
        # can easily be fed to a neural network.
        # The network expects a (height, width, 2) array with the last dimenstion
        # containing the visibility and adjacency values for the corresponding cell
        self.player_grid = numpy.zeros((height, width, Game.props_per_cell))

        # The networks won't need to flag anything, so we store this separately
        self.flag_grid = numpy.zeros((height, width))

        # Set as None since we generate it based on the first cell revealed
        self._true_grid = None

        if initial_moves != 0:
            self.initialise_grid(initial_cell=None, initial_moves=initial_moves)

    def __deepcopy__(self, memo):
        new = Game(self.width, self.height, self.number_of_mines)

        new.player_grid = numpy.copy(self.player_grid)
        new.flag_grid = numpy.copy(self.flag_grid)
        new._true_grid = deepcopy(self._true_grid)

        memo[id(self)] = new
        return new

    @classmethod
    def grid_size(cls, width: int, height: int):
        return width * height * cls.props_per_cell

    @property
    def player_won(self):
        return self.game_over and self.cells_hidden == self.number_of_mines

    @property
    def initialized(self):
        return self._true_grid is not None

    def neighbours(self, coords: Coords):
        row_range = range(max(0, coords.row - 1), min(self.height, coords.row + 2))
        col_range = range(max(0, coords.col - 1), min(self.width, coords.col + 2))
        return (
            Coords(r, c)
            for r, c in product(row_range, col_range)
            if r != coords.row or c != coords.col
        )

    def initialise_grid(self, initial_cell: Coords = None, initial_moves: int = 0):
        """Setup the grid so that the first cell has desirable properties"""

        all_cells = [
            Coords(r, c) for r, c in product(range(self.height), range(self.width))
        ]

        if initial_cell is None:
            initial_cell = random.choice(all_cells)

        def allowed(cell):
            # We don't allow mines to be on or adjacent to the initial cell
            return initial_cell != cell and not initial_cell.adjacent(cell)

        # Pick mines from allowed cells
        mines = random.sample(list(filter(allowed, all_cells)), self.number_of_mines)

        # The true grid can be something easier to work with
        self._true_grid = [[0 for c in range(self.width)] for r in range(self.height)]

        # populate adjacency values, then set mine positions
        for mine in mines:
            for r, c in self.neighbours(mine):
                self._true_grid[r][c] += 1

        for r, c in mines:
            self._true_grid[r][c] = Game.MINE

        if initial_moves > 0:
            safe_cells = [cell for cell in all_cells if cell not in mines]
            moves = random.sample(safe_cells, initial_moves)
            for move in moves:
                self.process_move(move)

    def flag_cell(self, coords: Coords):
        """Toggles the flag at the specified cell"""

        visible = self.player_grid[coords][Game.VISIBLE_INDEX]
        if visible:
            # cell is visible, nothing to do
            return True

        flagged = self.flag_grid[coords] == 1

        self.flag_grid[coords] = 0 if flagged else 1
        self.number_of_flags += -1 if flagged else 1

    def reveal_cell(self, coords: Coords):
        """Reveal the specified cell, and all adjacent cells if none contain mines"""

        cell = self.player_grid[coords]

        visible = cell[Game.VISIBLE_INDEX]
        flagged = self.flag_grid[coords]

        if visible or flagged:
            # cell is visible or flagged, nothing to do
            return

        true_value = self._true_grid[coords.row][coords.col]
        if true_value == Game.MINE:
            # Unlucky...
            self.game_over = True
            return

        self.cells_hidden -= 1
        cell[Game.VISIBLE_INDEX] = 1
        cell[Game.ADJACENT_INDEX] = true_value / 8

        if true_value == 0:
            # Reveal neighbours if no adjacent mines
            for other in self.neighbours(coords):
                self.reveal_cell(other)

    def process_move(self, coords: Coords, toggle_flag: bool=False):
        """Process a single move.
        A move is either revealing a cell or toggling a flag on a cell.
        """

        if not self.initialized:
            # First move
            self.initialise_grid(coords)
            self.reveal_cell(coords)
        elif toggle_flag:
            self.flag_cell(coords)
        else:
            self.reveal_cell(coords)

        if self.cells_hidden == self.number_of_mines:
            # Victory!
            self.game_over = True

    def valid_moves(self):
        return [
            Coords(r, c)
            for r in range(self.height)
            for c in range(self.width)
            if self.player_grid[r, c, Game.VISIBLE_INDEX] == 0
        ]

    def _random_safe_tiles(self, num_tiles: int):
        """Generate the coordinates of a random, safe tile"""

        if not self.initialized:
            return Coords(
                random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            )
        else:
            safe_tiles = [
                Coords(r, c)
                for r in range(self.height)
                for c in range(self.width)
                if self.player_grid[r, c, Game.VISIBLE_INDEX] == 0
                and self._true_grid[r][c] != Game.MINE
            ]
            return random.sample(safe_tiles, num_tiles)

    def print(self):
        """Prints the game's board.
        TODO: Cleanup
        """

        # Print top column letters
        top_label = "     " + "".join(f"{c+1:3} " for c in range(self.width))
        horizontal = "   " + (4 * self.width * "-") + "-"

        print(top_label)
        print(horizontal)

        for r in range(self.height):
            row_string = f"{r+1:3} |"
            for c in range(self.width):
                visible = self.player_grid[r, c, Game.VISIBLE_INDEX] == 1
                if visible or self.game_over:
                    row_string += f" {self._true_grid[r][c]} |"
                else:
                    is_flagged = self.flag_grid[r, c] == 1
                    row_string += " F |" if is_flagged else "   |"

            print(row_string)
            print(horizontal)

        print("")


def play_again():
    choice = input("Play again? (y/n): ")

    return choice.lower() == "y"


def get_move(width: int, height: int, mines_left: int):
    usage = (
        'Type the cell as <row>,<column> (eg. "3,4").\n'
        'To toggle a flag, add "f" after the cell coordinates (eg. "3,5 f").'
    )

    pattern = fr"([0-9]+),\s*([0-9]+)(\s+f)?"
    while True:
        user_input = input(f"Enter the cell ({mines_left} mines left): ")
        match = re.match(pattern, user_input)

        if user_input == "help":
            print(usage)
        elif match:
            row = int(match.group(1)) - 1
            col = int(match.group(2)) - 1
            toggle_flag = bool(match.group(3))

            if 0 <= row < height and 0 <= col < width:
                return Coords(row, col), toggle_flag
            else:
                print("Invalid coordinates")
        else:
            print("Invalid Input.")
            print(usage)


def play_game(width: int = 10, height: int = 10, number_of_mines: int = 10):

    while True:

        game = Game(width, height, number_of_mines)

        start_time = time.time()

        while True:
            mines_left = game.number_of_mines - game.number_of_flags

            game.print()

            if game.game_over:
                print("Game Over")
                if game.number_of_mines == game.cells_hidden:
                    minutes, seconds = divmod(int(time.time() - start_time), 60)
                    print(
                        "You Win. "
                        f"It took you {minutes} minutes and {seconds} seconds."
                    )
                break

            coords, toggle_flag = get_move(game.width, game.height, mines_left)

            game.process_move(coords, toggle_flag)

        if play_again():
            game = Game(width, height, number_of_mines)
        else:
            break
