## 5 fold cross validation with early stopping
/!\ validating on the test set /!\
Average metric over the 5 folds + standard deviation (i.e. we select the best epoch based on the validation accuracy for each fold)
Every metric is for the validation set if not specified otherwise

 learning_rate |hidden_size|num_layers|bidirectional|dropout|TRAIN accuracy | accuracy | Se | Sp | PPV | NPV
--|--|--|--|--|--|--|--|--|--|--
0.001 | 100 | 1 | False | 0.0 | 0.62 (+ 0.04) | 0.60 (+ 0.05) | 0.63 (+ 0.27) | 0.56 (+ 0.35) | 0.68 (+ 0.18) | 0.50 (+ 0.26)
