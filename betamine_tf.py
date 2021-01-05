import tensorflow as tf
import numpy
from minesweeper_tf import Game, Coords
import random
from Game_Generation import create_states

game_options = {
    "width": 4,
    "height": 4,
    "number_of_mines": 2,
}

convolutionlayer_model = tf.keras.Sequential(
    [
        # tf.keras.layers.Reshape((10,10,1)),
        tf.keras.layers.Conv2D(filters= 9, kernel_size=3),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(game_options["width"] * game_options["height"] , activation=tf.keras.activations.softmax)
    ])

convolutionlayer_model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.2),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = ['accuracy']
)

training_states = create_states(number_of_games = 1000, **game_options)
evaluation_states = create_states(number_of_games = 100, **game_options)

# convolutionlayer_model.predict(training_states[0][0].transpose())

convolutionlayer_model.fit( training_states[0] , training_states[1], batch_size = 64, epochs = 10)

print('Test accuracy:', convolutionlayer_model.evaluate(evaluation_states[0], evaluation_states[1], batch_size = 64)[1] )


