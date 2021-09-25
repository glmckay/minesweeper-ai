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
To create a model with parameters with the highest successrate (these data are all stored in Results/TrainingResults.csv), input
>model = best_model()


# Creating a random state
To create a random state, input
>state = create_state()

# Model Prediction
To have the model guess where the mines are, and to see where the model would play, input
>guess_mine_location(model,state)
