Work by Paul Lerner under Laurence Likforman-Sulem supervision.
# Main
arguments should be the hyperparameters :
- is_lstm
- learning_rate
- hidden_size
- num_layers
- bidirectional
- dropout
- clip
- task_i
- window_size

in that specific order, boolean arguments are not case sensitive, cf. `str2bool()`.  
to set any `task_i` or `window_size` to `None` non-numerical will do. e.g :  
`python main.py t 1e-3 10 4 t 0.0 5.0 0 none`  
If you want to change any other hyperparameters you will have to dive into the `main.py` code.

# Dependencies
- PyTorch
- Sci-kit learn
- NumPy

# Utils
`utils.py` contains misc. utilitaries such as measures and task dictionaries, confusion matrix.

# Data
## Loading
is done in `load_data.py`. It loads the subjetcs data and labels (1 for PD 0 for control) in the lists `data` and  `targets`, respectively.  
 It discards the subjects who didn't perform the spiral, i.e. Subjects 46 (control), 60 (PD) and 66 (control), counting from zero.

 The task sequence is in average **2286** &rarr; task duration is in average 11.4s  

task | duration (average) | duration (3rd quartile) | duration std | duration per letters
--|--|--|--|--
spiral | 2873 | 3117 |2242 | NA
l | 1668.01 | ?|724.14 | 333
le | 1984.16 |?| 999.61 | 198
les | 2305.01 |?| 1095.54 | 153
lektorka | 2608.48 | ?|1333.76  | 163
porovnat | 2315.08 | ?|1033.69 |144
nepopadnout | 1469.29 | ?|650.65 | 133
tram | 3086.13 |? |1158.81 | 146

## Data split
In order to provide for a meaningful comparison with works from Drotar et al. and Moetesum et al. we will evaluate our results using a 10-fold cross validation  
The split is done before *Training*

# Model
The code allows to choose between GRU and LSTM thanks to the `is_lstm` hyperparameter. The forget gate bias will be init to 1 only if the model is LSTM.
