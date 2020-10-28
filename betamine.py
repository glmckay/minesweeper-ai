from agent import agent
import numpy
import minesweeper

def init_agent():
	ai = agent.Network([10*10*3, 10*10*2])

def play_move(ai, curr_grid):
	grid_prob = ai.evaluate(curr_grid)
	index = numpy.argmax(grid_prob)
	position = index % (10*10)

	# position, flag_or_not
	return position / 10, position % 10, index / (10*10)

	# cost = (TrueBoard - flagBoard) + 0.1*moves -> delta