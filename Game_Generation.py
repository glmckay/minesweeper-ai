from minesweeper_tf import Game, Coords
import numpy
import random


def non_losing_indices(state):
    return [
        coords
        for coords in state.valid_moves()
        if state._true_grid[coords.row][coords.col] != Game.MINE
    ]


def move_from_output(state, output):
    valid_outputs = (
        (output[m.row * state.width + m.col], m) for m in state.valid_moves()
    )
    return min(valid_outputs)[1]


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


def create_test_state_pairs(**kwargs):
    width = kwargs["width"]
    height = kwargs["height"]
    number_of_games = kwargs["number_of_games"]
    number_of_mines = kwargs["number_of_mines"]

    game_states = numpy.ndarray(shape=(number_of_games, width, height, 1))
    game_solutions = numpy.ndarray(shape=(number_of_games, width * height))

    state_generator = create_states(**kwargs)

    for i, state in enumerate(state_generator):
        game_states[i] = state.player_grid[:, :, 1:]

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


def test_model(model, states):

    inputs = numpy.array([state.player_grid[:, :, 1:] for state in states])

    outputs = model.predict(inputs)

    successes = 0
    total = 0
    for state, output in zip(states, outputs):

        state.process_move(move_from_output(state, output))

        total += 1
        if not state.game_over:
            successes += 1
    print(
        f"Model did not game over in {successes} out of {len(states)} games "
        f"({successes / total * 100:.3f} %)"
    )


def test_against_game(model, number_of_games, game_args):
    wins = 0

    games = [Game(**game_args) for i in range(number_of_games)]

    while len(games) > 0:
        inputs = numpy.array([game.player_grid[:, :, 1:] for game in games])
        outputs = model(inputs)
        for game, output in zip(games, outputs):
            game.process_move(move_from_output(game, output))

            if game.player_won:
                wins += 1

        games = [game for game in games if not game.game_over]

    print(
        f"Model won {wins} out of {number_of_games} games "
        f"({wins / number_of_games * 100:.3f} %)"
    )
