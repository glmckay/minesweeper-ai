from Game_Generation import (
    create_states,
    create_test_state_pairs,
)
from numpy import save
import json
import pickle

# Initialize the game settings
game_options = {
    "width": 5,
    "height": 5,
    "number_of_mines": 4,
}

# Create a list of game states for training
training_states = create_test_state_pairs(number_of_games=300000, **game_options)

# Create a list of game states for evaluation (validation)
evaluation_states = list(create_states(number_of_games=10000, **game_options))

# Save them to files. We need two files, because together, it exceeds the 100Mb limit of github
with open("DataSets/TrainingSets1.pkl", 'wb') as train_file:
    pickle.dump([training_states[0][:150000],training_states[1][:150000]], train_file, pickle.HIGHEST_PROTOCOL)
with open("DataSets/TrainingSets2.pkl", 'wb') as train_file:
    pickle.dump([training_states[0][150000:],training_states[1][150000:]], train_file, pickle.HIGHEST_PROTOCOL)

with open("DataSets/EvalStates.pkl", 'wb') as eval_file:
    pickle.dump(evaluation_states, eval_file, pickle.HIGHEST_PROTOCOL)