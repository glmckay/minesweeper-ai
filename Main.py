from copy import deepcopy
import os

from tensorflow.python.ops.gen_math_ops import sigmoid
from Game_Generation import (
    create_states,
    create_test_state_pairs,
    test_model,
    test_against_game,
)
from numpy import load
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from minesweeper_tf import play_game
import Game_Generation
from betamine_model_testing import best_model , test_parameters_to_csv

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

# Initialize the game settings
game_options = {
    "width": 5,
    "height": 5,
    "number_of_mines": 4,
}

# Creates a random game state
def create_state():
    games = [(i,state) for i,state in enumerate(create_states(**game_options, number_of_games = 1))]
    state = games[0][1]
    state.print()
    return state


# A function that takes a model and a game state, then outputs a grid of where the model think the mine is and where it wants to move
def guess_mine_location(model,state):
    prediction = model.predict(np.array([state.player_grid]))
    Game_Generation.print_output(prediction)
    coord = Game_Generation.move_from_output(state,prediction[0])
    print(f"The model wants to play the coordinate {coord}.")


# A function that 
def plays_move(model,state):
    return 0
