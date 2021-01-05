from minesweeper_tf import Game, Coords
import numpy
import random


def non_losing_indices(state):
    return [
        coords
        for coords in state.valid_moves()
        if state._true_grid[coords.row][coords.col] != Game.MINE
    ]


def create_states(
    width: int = 10,
    height: int = 10,
    number_of_mines: int = 10,
    number_of_games: int = 10000,
):
    game_states = numpy.ndarray(shape=(number_of_games, width, height, 1))
    game_solutions = numpy.ndarray(shape=(number_of_games, width * height))

    for x in range(number_of_games):
        state = None
        while True: 
            state = Game(
                width, height, number_of_mines, initial_moves=random.randint(1, number_of_mines)
            )
            if not state.game_over:
                break

        game_states[x] = state.player_grid[:,:,1:]

        desired = numpy.zeros((width * height,))
        valid_non_mines = non_losing_indices(state)
        for coords in valid_non_mines:
            desired[coords.row * state.width + coords.col] = 1 / len(valid_non_mines)

        game_solutions[x] = desired

    return [game_states, game_solutions]
