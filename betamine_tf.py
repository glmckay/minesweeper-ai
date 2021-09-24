from copy import deepcopy
import os
from Game_Generation import (
    create_states,
    create_test_state_pairs,
    test_model,
    test_against_game,
)
from numpy import load
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 

# Initialize the game settings
game_options = {
    "width": 5,
    "height": 5,
    "number_of_mines": 4,
}

# # Create a list of game states for training
# training_states = create_test_state_pairs(number_of_games=300000, **game_options)

# # Create a list of game states for evaluation (validation)
# evaluation_states = list(create_states(number_of_games=10000, **game_options))


# Import the training and evaluation datasets
with open("DataSets/TrainingSets1.pkl", 'rb') as train_file:
    training_states1 = pickle.load(train_file)
with open("DataSets/TrainingSets2.pkl", 'rb') as train_file:
    training_states2 = pickle.load(train_file)

# Merge the two training sets together
training_states = [np.append(training_states1[0], training_states2[0], axis=0), np.append(training_states1[1], training_states2[1], axis=0)]

with open("DataSets/EvalStates.pkl", 'rb') as eval_file:
    evaluation_states = pickle.load(eval_file)


n = game_options["width"] * game_options["height"]

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(10 * n, activation=tf.keras.activations.softmax),
        #tf.keras.layers.Dense(10 * n, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(20 * n, activation=tf.keras.activations.softmax),
        tf.keras.layers.Dense(10 * n, activation=tf.keras.activations.softmax),
        tf.keras.layers.Dense(5 * n, activation=tf.keras.activations.softmax),
        # tf.keras.layers.Dense(n, activation=tf.keras.activations.softmax),
        # tf.keras.layers.Reshape((game_options["width"], game_options["height"], 1)),
        # tf.keras.layers.Conv2D(filters=9, kernel_size=3),
        # tf.keras.layers.MaxPool2D(pool_size=2),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            n,
            activation = None
            # tf.keras.activations.sigmoid,
            # tf.keras.activations.softmax,
        ),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=3),
    # loss=tf.keras.losses.CategoricalCrossentropy(),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics=["accuracy"],
)

model.fit(
    training_states[0], training_states[1], batch_size=60, epochs=5
)

# "Good" parameters:
#  CategoricalCrossentropy:
#    Batch size ~ 150, learning rate ~ 0.01
#  MeanSquaredError:
#    Batch size ~ 60, learning rate ~ 3

# Play one move from all the game states in evaluation_states and print the number of moves that did not hit a mine
test_model(model, deepcopy(evaluation_states))

# Play some number of full games (from start to finish) and print the winning percentage
test_against_game(model, 1000, game_options)
