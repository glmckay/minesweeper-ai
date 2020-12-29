from minesweeper import Game, Coords
import numpy
import random

def get_valid_indices(state):
    return [
        i
        for i in range(state.width * state.height)
        if state._player_grid[i * Game.props_per_cell + Game.VISIBLE_OFFSET] == 0
    ]

def create_states ( width: int = 10, height: int = 10, number_of_mines: int = 10, number_of_games: int = 10000):
    game_states = numpy.ndarray(shape = (number_of_games,width*height,1))
    game_solutions = numpy.ndarray(shape = (number_of_games,width*height))

    for x in range(number_of_games):
        state = Game(width , height , number_of_mines , initial_moves = random.randint(1,10) )
        numpy.append(game_states,state.player_grid)

        desired = numpy.zeros((1, width * height))
        valid_indices = get_valid_indices(state)
        non_losing_indices = [
            i
            for i in valid_indices
            if state._true_grid[i // state.width][i % state.height] != Game.MINE
        ]
        for i in non_losing_indices:
            desired[0][i] = 1 / len(non_losing_indices)

        numpy.append(game_solutions,desired)

    return [game_states, game_solutions]

