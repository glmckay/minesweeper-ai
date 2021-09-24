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
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 
import pandas as pd
from collections import defaultdict

""" Next Time:
    Convolution layer to improve behavior with more complicated games?
    Choosing better training examples? (idk, just writing ideas)
"""

# Initialize the game settings
game_options = {
    "width": 5,
    "height": 5,
    "number_of_mines": 4,
}

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

# a function that creates a model based on different activations
def create_model_list(activation1, activation2):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(20 * n, activation = activation1),
            tf.keras.layers.Dense(10 * n, activation = activation1),
            tf.keras.layers.Dense(5 * n, activation = activation1),
            tf.keras.layers.Dense(n,activation = activation2),
        ]
    )
    return model

# a function that compiles a given model based on a chosen learning rate, optimizer and loss function
def compile_model(model, learning_rate, optimizer, loss):
    model.compile(
        optimizer = optimizer(learning_rate = learning_rate),
        loss = loss,
        metrics=["accuracy"],
    )

# A function that creates a DataFrame containing the results of trying out various different parameters into the model
def test_models(activations1, activations2, learning_rates, optimizers, losses, batches, epochs):
    results = defaultdict(list)
    for activation1 in activations1:
        for activation2 in activations2:
            for learning_rate in learning_rates:
                for optimizer in optimizers:
                    for loss in losses:
                        for batch in batches:
                            for epoch in epochs:
                                model = create_model_list(activation1, activation2)
                                compile_model(model, learning_rate, optimizer, loss())
                                model.fit(training_states[0], training_states[1], batch_size = batch, epochs = epoch)
                                success_rate =  test_model(model, deepcopy(evaluation_states),1)
                                win_rate = test_against_game(model, 1000, game_options,1)

                                results["Activation1"].append(activation1.__name__)
                                results["Activation2"].append("None" if activation2 == None else activation2.__name__)
                                results["LearningRate"].append(learning_rate)
                                results["Optimizer"].append(optimizer.__name__)
                                results["Loss"].append(loss.__name__)
                                results["Batch"].append(batch)
                                results["Epoch"].append(epoch)
                                results["SuccessRate"].append(success_rate)
                                results["WinRate"].append(win_rate)
    return pd.DataFrame(results)


# Initialize the different parameters to test
test_parameters = {"activations1": [tf.keras.activations.tanh], 
    "activations2": [tf.keras.activations.sigmoid], 
    "learning_rates": [1], 
    "optimizers": [tf.keras.optimizers.SGD], 
    "losses": [tf.keras.losses.CategoricalCrossentropy], 
    "batches": [30], 
    "epochs": [3]
    }

test_results = test_models(**test_parameters)

# add the results to previously computed ones
previous_result = pd.read_csv("Results/TrainingResults.csv")
test_results = pd.concat([test_results,previous_result]).drop_duplicates().reset_index(drop=True)

# save the results
test_results.to_csv("Results/TrainingResults.csv", index=False)

