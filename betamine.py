from agent import agent
import numpy
import minesweeper


width = 10
height = 10


def init_agent():
    ai = agent.Network([width * height * 3, width * height * 2])


def play_move(ai, curr_grid):
    grid_prob = ai.evaluate(curr_grid)
    index = numpy.argmax(grid_prob)

    position, toggle_flag = divmod(index, width * height)
    return minesweeper.Coords(*divmod(position, width)), toggle_flag

    # cost = (TrueBoard - flagBoard) + 0.1*moves -> delta

