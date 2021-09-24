from minesweeper_tf import Game, Coords
import numpy
import random

# function that returns all the coordinates that do not contain mines
def non_losing_indices(state):
    return [
        coords
        for coords in state.valid_moves()
        if state._true_grid[coords.row][coords.col] != Game.MINE
    ]


# Convenience function to print the output in a readable way
def print_output(output):
    for i in range(5):
        for j in range(5):
            print(f"{output[0][i*5+j]:.5f}", end=" ")
        print()


def move_from_output(state, output):
    valid_outputs = (
        (output[m.row * state.width + m.col], m) for m in state.valid_moves()
    )
    return min(valid_outputs)[1]

# generate some numbers of game states (games with some random number of moves already played)
def create_states(
    width: int = 10,
    height: int = 10,
    number_of_mines: int = 10,
    initial_move_weights=None,
    number_of_games: int = 10000,
):
    if initial_move_weights is None:
        initial_move_weights = [1] * number_of_mines

    for x in range(number_of_games):
        state = None
        initial_moves = random.choices(
            range(1, len(initial_move_weights) + 1), initial_move_weights, k=1
        )[0]
        while True:
            state = Game(
                width,
                height,
                number_of_mines,
                initial_moves,
            )
            if not state.game_over:
                break
        yield state

# create multiple pairs of (game state, game_solutions)
def create_test_state_pairs(**kwargs):
    width = kwargs["width"]
    height = kwargs["height"]
    number_of_games = kwargs["number_of_games"]
    number_of_mines = kwargs["number_of_mines"]

    game_states = numpy.ndarray(shape=(number_of_games, width, height, 2))
    game_solutions = numpy.ndarray(shape=(number_of_games, width * height))

    state_generator = create_states(**kwargs)

    for i, state in enumerate(state_generator):
        game_states[i] = state.player_grid

        desired = numpy.zeros((width * height,))
        for row in range(height):
            for col in range(width):
                if state._true_grid[row][col] == Game.MINE:
                    desired[row * width + col] = 1 / number_of_mines
        # valid_non_mines = non_losing_indices(state)
        # for coords in valid_non_mines:
        #     desired[coords.row * state.width + coords.col] = 1 / len(valid_non_mines)

        game_solutions[i] = desired

    return [game_states, game_solutions]

# calculates the success rate of the model (number of times it does not pick a mine at a particular game state)
def test_model(model, states, to_print = 0):

    inputs = numpy.array([state.player_grid for state in states])

    outputs = model.predict(inputs)

    successes = 0
    total = 0
    for state, output in zip(states, outputs):

        state.process_move(move_from_output(state, output))

        total += 1
        if not state.game_over:
            successes += 1
    if to_print == 0:
        print(
            f"Model did not game over in {successes} out of {len(states)} states "
            f"({successes / total * 100:.3f} %)"
        )
    else:
        return successes / total * 100

# finds the win rate of the model, that is the probability that the model finishes a full game
def test_against_game(model, number_of_games, game_args, to_print = 0):
    wins = 0

    games = [Game(**game_args) for i in range(number_of_games)]

    while len(games) > 0:
        inputs = numpy.array([game.player_grid for game in games])
        outputs = model(inputs)
        for game, output in zip(games, outputs):
            game.process_move(move_from_output(game, output))

            if game.player_won:
                wins += 1

        games = [game for game in games if not game.game_over]

    if to_print == 0:
        print(
            f"Model won {wins} out of {number_of_games} games "
            f"({wins / number_of_games * 100:.3f} %)"
        )
    else:
        return wins / number_of_games * 100
