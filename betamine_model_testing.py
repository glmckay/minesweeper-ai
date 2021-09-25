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
import pandas as pd
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

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
def create_model_list(layers):
    sequence = [tf.keras.layers.Flatten()]
    for layer_type,layer_attributes in layers:
        sequence.append(layer_type(**layer_attributes))
    model = tf.keras.Sequential(sequence)
    return model

# a function that compiles a given model based on a chosen learning rate, optimizer and loss function
def compile_model(model, learning_rate, optimizer, loss):
    model.compile(
        optimizer = optimizer(learning_rate = learning_rate),
        loss = loss,
        metrics=["accuracy"],
    )

# A function that creates a DataFrame containing the results of trying out various different parameters into the model
def test_models(layers_struct, learning_rates, optimizers, losses, batches, epochs):
    results = defaultdict(list)
    for layers in layers_struct:
        for learning_rate in learning_rates:
            for optimizer in optimizers:
                for loss in losses:
                    for batch in batches:
                        for epoch in epochs:
                            print(f"Creating model with layers {layers}")
                            model = create_model_list(layers)

                            print(f"compiling with loss {loss}, optimizer {optimizer} and learning rate {learning_rate}")
                            compile_model(model, learning_rate, optimizer, loss())

                            print(f"Fitting model with batch size {batch} in {epoch} epochs")
                            model.fit(training_states[0], training_states[1], batch_size = batch, epochs = epoch)
                            success_rate =  test_model(model, deepcopy(evaluation_states), 1)
                            win_rate = test_against_game(model, 1000, game_options, 1)

                            for i in range(len(layers)):
                                layer_type, layer_attributes = layers[i]
                                results[f"Layer{i+1}_Name"].append(layer_type.__name__)
                                for elem in layer_attributes:
                                    if layer_attributes[elem] == None:
                                        results[f"Layer{i+1}_{elem}"].append("None")
                                    elif callable(layer_attributes[elem]):
                                        results[f"Layer{i+1}_{elem}"].append(layer_attributes[elem].__name__)
                                    else:
                                        results[f"Layer{i+1}_{elem}"].append(layer_attributes[elem])
                            results["LearningRate"].append(learning_rate)
                            results["Optimizer"].append(optimizer.__name__)
                            results["Loss"].append(loss.__name__)
                            results["Batch"].append(batch)
                            results["Epoch"].append(epoch)
                            results["SuccessRate"].append(success_rate)
                            results["WinRate"].append(win_rate)
    return pd.DataFrame(results)


# Initialize the different parameters to test. Note, for layer structure, all of them must have the same number of layers and layer attributes (or else we can't construct the data frame because of different column size)
test_parameters = {
    "layers_struct": [
        [
            [tf.keras.layers.Dense, {"units": 20*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 10*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 5*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 1*n, "activation": None}]
        ],
        [
            [tf.keras.layers.Dense, {"units": 20*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 10*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 5*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 1*n, "activation": tf.keras.activations.sigmoid}]
        ],
        [
            [tf.keras.layers.Dense, {"units": 20*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 10*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 5*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 1*n, "activation": tf.keras.activations.softmax}]
        ]
    ],
    "learning_rates": [1, 0.1, 0.01],
    "optimizers": [tf.keras.optimizers.SGD], 
    "losses": [tf.keras.losses.MeanSquaredError, tf.keras.losses.CategoricalCrossentropy],
    "batches": [30, 100],
    "epochs": [3, 5]
}

test_results = test_models(**test_parameters)

# add the results to previously computed ones
previous_result = pd.read_csv("Results/TrainingResults.csv")
test_results = pd.concat([test_results, previous_result]).drop_duplicates().reset_index(drop=True)

# save the results
test_results.to_csv("Results/TrainingResults.csv", index=False)

