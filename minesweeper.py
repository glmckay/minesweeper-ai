"""A command line version of Minesweeper"""

import random
import re
import time
from string import ascii_lowercase


def setup_grid(grid_size, start, number_of_mines):
    empty_grid = [["0" for i in range(grid_size)] for i in range(grid_size)]

    mines = get_mines(empty_grid, start, number_of_mines)

    for i, j in mines:
        empty_grid[i][j] = "X"

    grid = get_numbers(empty_grid)

    return (grid, mines)


def show_grid(grid):
    grid_size = len(grid)

    horizontal = "   " + (4 * grid_size * "-") + "-"

    # Print top column letters
    top_label = "     "

    for i in ascii_lowercase[:grid_size]:
        top_label = top_label + i + "   "

    print(top_label + "\n" + horizontal)

    # Print left row numbers
    for idx, i in enumerate(grid):
        row = "{0:2} |".format(idx + 1)

        for j in i:
            row = row + " " + j + " |"

        print(row + "\n" + horizontal)

    print("")


def get_random_cell(grid):
    grid_size = len(grid)

    a = random.randint(0, grid_size - 1)
    b = random.randint(0, grid_size - 1)

    return (a, b)


def getneighbors(grid, rowno, colno):
    gridsize = len(grid)
    neighbors = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            elif -1 < (rowno + i) < gridsize and -1 < (colno + j) < gridsize:
                neighbors.append((rowno + i, colno + j))

    return neighbors


def get_mines(grid, start, numberofmines):
    mines = []
    neighbors = getneighbors(grid, *start)

    for i in range(numberofmines):
        cell = get_random_cell(grid)
        while cell == start or cell in mines or cell in neighbors:
            cell = get_random_cell(grid)
        mines.append(cell)

    return mines


def get_numbers(grid):
    for rowno, row in enumerate(grid):
        for colno, cell in enumerate(row):
            if cell != "X":
                # Gets the values of the neighbors
                values = [grid[r][c] for r, c in getneighbors(grid, rowno, colno)]

                # Counts how many are mines
                grid[rowno][colno] = str(values.count("X"))

    return grid


def show_cells(grid, curr_grid, row, col):
    # Exit function if the cell was already shown
    if curr_grid[row][col] != " ":
        return

    # Show current cell
    curr_grid[row][col] = grid[row][col]

    # Get the neighbors if the cell is empty
    if grid[row][col] == "0":
        for r, c in getneighbors(grid, row, col):
            # Repeat function for each neighbor that doesn't have a flag
            if curr_grid[r][c] != "F":
                show_cells(grid, curr_grid, r, c)


def play_again():
    choice = input("Play again? (y/n): ")

    return choice.lower() == "y"


def parse_input(input_string, grid_size, help_message):
    cell = ()
    flag = False
    message = "Invalid cell. " + help_message

    pattern = fr"([0-9]+),\s+([0-9]+)(\s+f)?"
    valid_input = re.match(pattern, input_string)

    if input_string == "help":
        message = help_message
    elif valid_input:
        row = int(valid_input.group(1)) - 1
        col = int(valid_input.group(2)) - 1
        flag = bool(valid_input.group(3))

        if -1 < row < grid_size:
            cell = (row, col)
            message = ""

    return {"cell": cell, "flag": flag, "message": message}


def play_game():
    grid_size = 10
    number_of_mines = 10


def playgame(grid_size=10, number_of_mines=10):
    curr_grid = [[" " for i in range(grid_size)] for i in range(grid_size)]

    grid = []
    flags = []
    start_time = 0

    help_message = (
        'Type the cell as <row>,<column> (eg. "3,4").'
        'To toggle a flag, add "f" after the cell coordinates (eg. "3,5 f").'
    )

    show_grid(curr_grid)
    print(help_message + " Type 'help' to show this message again.\n")

    while True:
        mines_left = number_of_mines - len(flags)
        prompt = input("Enter the cell ({} mines left): ".format(mines_left))
        result = parse_input(prompt, grid_size, help_message + "\n")

        message = result["message"]
        cell = result["cell"]

        if cell:
            print("\n\n")
            row, col = cell
            curr_cell = curr_grid[row][col]
            flag = result["flag"]

            if not grid:
                grid, mines = setup_grid(grid_size, cell, number_of_mines)
            if not start_time:
                start_time = time.time()

            if flag:
                # Add a flag if the cell is empty
                if curr_cell == " ":
                    curr_grid[row][col] = "F"
                    flags.append(cell)
                # Remove the flag if there is one
                elif curr_cell == "F":
                    curr_grid[row][col] = " "
                    flags.remove(cell)
                else:
                    message = "Cannot put a flag there"

            # If there is a flag there, show a message
            elif cell in flags:
                message = "There is a flag there"

            elif grid[row][col] == "X":
                print("Game Over\n")
                show_grid(grid)
                if play_again():
                    play_game()
                return

            elif curr_cell == " ":
                show_cells(grid, curr_grid, row, col)

            else:
                message = "That cell is already shown"

            if set(flags) == set(mines):
                minutes, seconds = divmod(int(time.time() - start_time), 60)
                print(
                    "You Win. "
                    "It took you {} minutes and {} seconds.\n".format(minutes, seconds)
                )
                show_grid(grid)
                if play_again():
                    play_game()
                return

        show_grid(curr_grid)
        print(message)


play_game()
