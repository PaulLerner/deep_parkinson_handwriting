Work by Paul Lerner under Laurence Likforman-Sulem supervision.

# Dependencies
- PyTorch
- Sci-kit learn
- NumPy

# Utils
`utils.py` contains misc. utilitaries such as measures and task dictionaries, confusion matrix, display functions (e.g. display 10-CV results).

# Data
## Loading
is done in `load_data.py`. It loads the subjetcs data and labels (1 for PD 0 for control) in the lists `data` and  `targets`, respectively.  
 It discards the subjects who didn't perform the spiral, i.e. Subjects 46 (control), 60 (PD) and 66 (control), counting from zero.
 
# Model
The code allows to choose between GRU and LSTM thanks to the `is_lstm` hyperparameter. The forget gate bias will be init to 1 only if the model is LSTM.
