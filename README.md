# minesweeper-ai
A minesweeper AI with neural networks. 

* Download the repository
* On the command line, go to the path of the download
* Type `< python -i Main.py>`

# To play a minesweeper game:
Type in the function 
>play_game()


or 
>play_game(width= , height= , number_of_mines= )

The default is currently set to width = 5, height = 5, number_of_mines= 6. 

To play the game, simply enter: `<row,column>` (eg. `<3,4>`) or to toggle a flag, type `<row, column f>` (eg. `<3,4 f>`). 


# Model Creation
To create a model, run 
>model = csv_to_model(**parameter)

where a sample parameter is the following:
```Python
test_parameters = {
    "layers_struct": [[
            [tf.keras.layers.Dense, {"units": 20*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 10*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 5*n, "activation":  tf.keras.activations.relu}],
            [tf.keras.layers.Dense, {"units": 1*n, "activation": tf.keras.activations.sigmoid}]]],
    "learning_rates": [1],
    "optimizers": [tf.keras.optimizers.SGD], 
    "losses": [tf.keras.losses.CategoricalCrossentropy],
    "batches": [100],
    "epochs": [10]
}
```
To create a model with parameters that give the highest successrate (out of all the parameters that we tested and stored in Results/TrainingResults.csv), run
>model = best_model()


# Creating a random state
To create a random state, input
>state = create_state()

# Model Prediction
To have the model guess where the mines are, and to see where the model would play, input
>guess_mine_location(model,state)
