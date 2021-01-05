from minesweeper import Game, Coords
import numpy
import random


def non_losing_indices(state):
    return [
        coords
        for coords in state.valid_moves()
        if state._true_grid[coords.r][coords.c] != Game.MINE
    ]


def create_states(
    width: int = 10,
    height: int = 10,
    number_of_mines: int = 10,
    number_of_games: int = 10000,
):
    game_states = numpy.ndarray(shape=(number_of_games, width * height, 1))
    game_solutions = numpy.ndarray(shape=(number_of_games, width * height))

    for x in range(number_of_games):
        state = Game(
            width, height, number_of_mines, initial_moves=random.randint(1, 10)
        )
        numpy.append(game_states, state.player_grid)

        desired = numpy.zeros((1, width * height))
        valid_non_mines = non_losing_indices(state)
        for i in valid_non_mines:
            desired[0][i] = 1 / len(valid_non_mines)

        numpy.append(game_solutions, desired)

    return [game_states, game_solutions]
