import component.agent as agent
from minesweeper import Game, Coords
import numpy
import random

game_options = {
    "width": 5,
    "height": 5,
    "number_of_mines": 3,
}


def init_agent(width: int, height: int, **kwargs):
    move_space_size = width * height * 2

    return agent.Network([Game.grid_size(width, height), move_space_size])


network = init_agent(**game_options)

for i in range(10000):
    game = Game(**game_options)    

    while not game.game_over:
        # game.print()
        grid_prob = network.evaluate(game.player_grid)

        #index = numpy.argmax(grid_prob[-1])

        grid_size = game.width * game.height
        valid_tiles = [i for i in range(grid_size) if game._player_grid[i * 3] == 0]
        valid_indices = valid_tiles #+ [i + grid_size for i in valid_tiles]

        index = random.choices(
            valid_indices,
            weights= [grid_prob[-1][0][x] for x in valid_indices]
        )[0]

        toggle_flag, position = divmod(index, game.width * game.height)

        # print(f"PLaying move: {divmod(position, game.width)}" + ("+flag" if toggle_flag else ""))
        game.process_move(Coords(*divmod(position, game.width)), toggle_flag)


    # game.print()

    if game.player_won:
        # print("WOW! The network actually won!")
        pass
    else:
        # print("unlucky")
        cost = numpy.zeros((1,game.width*game.height*2))
        cost[0][index] = 1
        delta = grid_prob[-1] - cost
        network.updateWeights(grid_prob,delta)



# game is over, spank network for losing
# cost = (TrueBoard - flagBoard) + 0.1*moves -> delta

def play():
    game = Game(**game_options)

    while not game.game_over:
        game.print()

        grid_prob = network.evaluate(game.player_grid)
        grid_size = game.width * game.height
        valid_tiles = [i for i in range(grid_size) if game._player_grid[i * 3] == 0]
        valid_indices = valid_tiles #+ [i + grid_size for i in valid_tiles]

        index = random.choices(
            valid_indices,
            weights= [grid_prob[-1][0][x] for x in valid_indices]
        )[0]

        toggle_flag, position = divmod(index, game.width * game.height)

        print(f"PLaying move: {divmod(position, game.width)}" + ("+flag" if toggle_flag else ""))
        game.process_move(Coords(*divmod(position, game.width)), toggle_flag)


    game.print()

    if game.player_won:
        print("WOW! The network actually won!")
    else:
        print("unlucky")