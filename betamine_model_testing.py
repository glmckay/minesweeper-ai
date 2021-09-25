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

                            print(f"Compiling with loss {loss.__name__}, optimizer {optimizer.__name__} and learning rate {learning_rate}")
                            compile_model(model, learning_rate, optimizer, loss())

                            print(f"Fitting model with batch size {batch} in {epoch} epochs")
                            model.fit(training_states[0], training_states[1], batch_size = batch, epochs = epoch)
                            success_rate =  test_model(model, deepcopy(evaluation_states))
                            win_rate = test_against_game(model, 1000, game_options)

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


# Sample format for the different parameters to test for the test_models function above. Note, for layer structure, all of them must have the same number of layers and layer attributes (or else we can't construct the data frame because of different column size)
""" test_parameters = {
    "layers_struct": [
        [
            [tf.keras.layers.Dense, {"units": 20*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 10*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 5*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 1*n, "activation": tf.keras.activations.sigmoid}]
        ]
    ],
    "learning_rates": [1],
    "optimizers": [tf.keras.optimizers.SGD], 
    "losses": [tf.keras.losses.CategoricalCrossentropy],
    "batches": [100],
    "epochs": [10]
} """

# A function that calls on generates the results from varies test parameters and stores the results in a csv file
def test_model_parameters(test_parameters):
    test_results = test_models(**test_parameters)

    # add the results to previously computed ones
    previous_result = pd.read_csv("Results/TrainingResults.csv")

    test_results = pd.concat([test_results, previous_result])

    subset = list(test_results.columns)
    subset.remove("SuccessRate")
    subset.remove("WinRate")
    test_results = test_results.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)

    # save the results
    test_results.to_csv("Results/TrainingResults.csv", index=False)


# A function that reads in a row from the TrainingResults.csv file, and outputs the corresponding model
# There must be a better way to input the layer type than how it's done here. However, tf.keras.layers is not callable...
def csv_to_model(row):
    layer_type = {
        "Dense": tf.keras.layers.Dense
    }
    Optimize = {
        "SGD": tf.keras.optimizers.SGD,
        "Adam": tf.keras.optimizers.Adam
    }
    layers = []
    k = 1
    # reconstruct all the parameters for the layers
    while f"Layer{k}_Name" in row.index and row[f"Layer{k}_Name"] != None:
        columns = [col for col in row.index if col.startswith(f"Layer{k}")]
        length = len(f"Layer{k}_")
        parameters = {}
        for attribute in columns:
            param = attribute[length:]
            if param == "Name" or row[attribute] == None:
                continue
            else:
                if row[attribute] == "None":
                    parameters[param] = tf.keras.activations.sigmoid
                elif row[attribute] == "relu":
                    parameters[param] = tf.keras.activations.relu
                else:
                    parameters[param] = row[attribute]
        layers.append([layer_type[row[f"Layer{k}_Name"]] , parameters])
        k += 1
    print(layers)
    model = create_model_list(layers)
    compile_model(model, row["LearningRate"], Optimize[row["Optimizer"]], loss = row["Loss"])
    model.fit(training_states[0], training_states[1], batch_size= row["Batch"], epochs=row["Epoch"])
    return model
 

# A function that outputs the the model with the highest success rate in the TrainingResults.csv file
def best_model():
    # import the previously computed results
    results = pd.read_csv("Results/TrainingResults.csv")
    
    row = results.iloc[results["SuccessRate"].idxmax()]
    print(row)
    model = csv_to_model(row)

    test_model(model, deepcopy(evaluation_states))
    test_against_game(model, 1000, game_options)

    return model

