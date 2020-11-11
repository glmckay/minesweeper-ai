import component.agent as agent
from minesweeper import Game, Coords
import numpy
import random


game_options = {
    "width": 5,
    "height": 5,
    "number_of_mines": 6,
}


def init_agent(width: int, height: int, **kwargs):
    move_space_size = width * height * 2

    return agent.Network(
        [
            Game.grid_size(width, height),
            move_space_size,
        ]
    )


def play(network, training=False, show_output=True):
    game = Game(**game_options)

    while not game.game_over:
        if show_output:
            game.print()

        grid_prob = network.evaluate(game.player_grid)
        valid_indices = [
            i
            for i in range(game.width * game.height)
            if game._player_grid[i * Game.props_per_cell + Game.VISIBLE_OFFSET] == 0
        ]

        index = max(valid_indices, key=lambda x: grid_prob[-1][0][x])

        # index = random.choices(
        #     valid_indices, weights=[grid_prob[-1][0][x] for x in valid_indices]
        # )[0]

        toggle_flag, position = divmod(index, game.width * game.height)

        if show_output:
            print(
                f"PLaying move: {tuple(x+1 for x in divmod(position, game.width))}"
                + ("+flag" if toggle_flag else "")
            )

        game.process_move(Coords(*divmod(position, game.width)), toggle_flag)

    if show_output:
        game.print()

        if game.player_won:
            print("WOW! The network actually won!")
        else:
            print("unlucky")

    if training:
        desired = numpy.zeros((1, game.width * game.height * 2))
        if game.player_won:
            desired[0][index] = 1
        else:
            non_losing_indices = [
                i
                for i in valid_indices
                if game._true_grid[i // game.width][i % game.height] != Game.MINE
            ]
            for i in non_losing_indices:
                desired[0][i] = 1 / len(non_losing_indices)

        delta = grid_prob[-1] - desired
        network.updateWeights(grid_prob, delta)

    return game.player_won


def win_rate(network):
    NUM_GAMES = 10000
    return (
        sum(1 for i in range(NUM_GAMES) if play(network, show_output=False)) / NUM_GAMES
    )


network = init_agent(**game_options)

print(f"Network win rate before training: {win_rate(network) * 100:.1f}%")

for i in range(100000):
    play(network, training=True, show_output=False)

print(f"Network win rate after training: {win_rate(network) * 100:.1f}%")