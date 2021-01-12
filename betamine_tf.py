from copy import deepcopy
import tensorflow as tf
from Game_Generation import create_states, create_test_state_pairs, test_model

game_options = {
    "width": 5,
    "height": 5,
    "number_of_mines": 3,
}

n = game_options["width"] * game_options["height"]

convolutionlayer_model = tf.keras.Sequential(
    [
        # tf.keras.layers.Reshape((10,10,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10*n),
        tf.keras.layers.Dense(20*n),
        tf.keras.layers.Dense(10*n),
        tf.keras.layers.Dense(5*n),
        # tf.keras.layers.Conv2D(filters=9, kernel_size=4),
        # tf.keras.layers.MaxPool2D(pool_size=2),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            game_options["width"] * game_options["height"],
            # activation=tf.keras.activations.softmax,
            activation=tf.keras.activations.sigmoid,
        ),
    ]
)

convolutionlayer_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=3),
    # loss=tf.keras.losses.CategoricalCrossentropy(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["accuracy"],
)

training_states = create_test_state_pairs(number_of_games=300000, **game_options)
evaluation_states = list(create_states(number_of_games=10000, **game_options))

# convolutionlayer_model.predict(training_states[0][0].transpose())

print()
print()

test_model(convolutionlayer_model, deepcopy(evaluation_states))

convolutionlayer_model.fit(
    training_states[0], training_states[1], batch_size=64, epochs=30
)

test_model(convolutionlayer_model, deepcopy(evaluation_states))

# print('Test accuracy:', convolutionlayer_model.evaluate(evaluation_states[0], evaluation_states[1], batch_size = 64)[1] )
