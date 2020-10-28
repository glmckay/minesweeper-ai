from agent import agent
import numpy
import minesweeper

def initAgent():
	agent = agent.Network([10*10, 10*10*2])

def playgame(weights, curr_grid):
	grid_prob = agent.evaluate(curr_grid)
	index = numpy.argmax(grid_prob)
	position = index % 10*10

	# position, flag_or_not
	return position / 10, position % 10, index / 10*10