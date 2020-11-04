import component.agent as agent
import numpy
from minesweeper import Game, Coords

import random

game_options = {
    "width": 10,
    "height": 10,
    "number_of_mines": 1,
}


def init_agent(width: int, height: int, **kwargs):
    move_space_size = width * height * 2

    return agent.Network([Game.grid_size(width, height), move_space_size])


network = init_agent(**game_options)
game = Game(**game_options)

while not game.game_over:
    game.print()

    grid_prob = network.evaluate(game.player_grid.transpose())
    # index = numpy.argmax(grid_prob[-1])
    index = random.choices(
        range(200),
        weights=grid_prob[-1].transpose()
    )[0]

    toggle_flag, position = divmod(index, game.width * game.height)

    print(f"PLaying move: {divmod(position, game.width)}" + ("+flag" if toggle_flag else ""))
    game.process_move(Coords(*divmod(position, game.width)), toggle_flag)


game.print()

if game.player_won:
    print("WOW! The network actually won!")
else:
    print("unlucky")
    cost = ""

# game is over, spank network for losing
# cost = (TrueBoard - flagBoard) + 0.1*moves -> delta

