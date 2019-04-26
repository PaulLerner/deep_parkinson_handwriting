#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""# Summary
- [Data](#Data)
- [Model](#Model)
- [Training](#Training)
    - [10-fold-cross-validation-(early-stopping)](#10-fold-cross-validation-(early-stopping))
- [Evaluation](#Evaluation)
- [Visualization](#Visualization)
    - [Interpretation](#Interpretation)
- [Implemented-but-not-used](#Implemented-but-not-used)
    - [Debug](#Debug)"""

# Dependencies

#visualization
import matplotlib.pyplot as plt
#math tools
import numpy as np

#machine learning
import torch
from sklearn.model_selection import StratifiedKFold
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device :\nin main :" ,device)
#utils
from time import time
import warnings

#to load and save lists
import pickle

#custom
from utils import *
from load_data import massage_data
from model import Model
from training import *

#to enable argument in command line
import sys

print( 'Argument List:', str(sys.argv[1:]))
print("arguments should be the hyperparameters :")
print("""is_lstm,
learning_rate,
hidden_size,
num_layers,
bidirectional,
dropout,
clip,
task_i,
window_size""")
print("in that specific order, boolean arguments are not case sensitive, cf. str2bool()")
## Hyperparameters

is_lstm=str2bool(sys.argv[1])
learning_rate = float(sys.argv[2])
hidden_size=int(sys.argv[3])
num_layers=int(sys.argv[4])
bidirectional=str2bool(sys.argv[5])
dropout=float(sys.argv[6])
try:
    clip=float(sys.argv[7])#clipping value to clip the gradients norm : set to None if you don't want to clip
except ValueError:
    clip=None
try:
    #set `task_i` to None if you want to train the model on all tasks at once (i.e. early fusion)
    #Else set `task_i` to the desired task index (cf. task2index)
    task_i=int(sys.argv[8])
except ValueError:
    task_i=None
try:
    #Set `window_size` to `None` if you don't want to split data into subsequence of fixed length
    window_size=int(sys.argv[9])
except ValueError:
    window_size=None

print("\nloading and massaging data, this might take a few seconds...")
data, targets= massage_data(task_i, compute_movement=False, downsampling_factor=1, window_size=window_size)
# Utils
#Cf `utils.py`

def return_results(train_metrics,valid_metrics,test_metrics,early_stopping,flat_falses):
    train_metrics,valid_metrics,test_metrics=np.asarray(train_metrics),np.asarray(valid_metrics),np.asarray(test_metrics)
    model_name="LSTM" if is_lstm else "GRU"
    task_name=index2task[task_i] if task_i is not None else str(task_i)
    results="{} ; {} ; {} ; True ; True ; {} ; {}  ; {} ; {} ; {} ; {} ; {} ; {} ; {:.2f} (+ {:.2f}) ; {:.2f} (+ {:.2f}) ; {:.2f} (+ {:.2f}) ".format(
    task_name,model_name,compute_movement,downsampling_factor,learning_rate,
        hidden_size,num_layers,bidirectional,carry_over,dropout,clip,
    np.mean(early_stopping),np.std(early_stopping),np.mean(train_metrics[:,1]),np.std(train_metrics[:,1]),
    np.mean(valid_metrics[:,1]),np.std(valid_metrics[:,1]))

    test_metrics=test_metrics.T
    for metric in test_metrics[1:]:#don't care about the loss
        mean,std=np.mean(metric),np.std(metric)
        results+="; {:.2f} (+ {:.2f}) ".format(mean,std)
    results+=" ; "
    results+=" ; ".join(map(str, flat_falses))
    return results

carry_over=0.0
if task_i is not None and window_size is None:#if we perform single task learning on the whole sequence
    input_size=len(data[0][0])#==7 if we take all the measures into account
else:
    input_size=len(data[0][0][0])#==7+8 if we train on multiple tasks, 7 if we train on subsequences (and single task)
bias=True
batch_first=False#we should unsqueeze(1) to add a batch dimension
batch_size=1
output_size=1#binary classif : 1 means PD, 0 means control

loss_fn=torch.nn.BCELoss()#Binary cross entropy

## 10 fold cross validation (early stopping)
#- set `run_CV` to `False` if you just want to train on one fold.
#- set `early_stopping` to `False` if you just want to train on a fixed n° of epochs.

#split in train valid and test set
skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_generator=skf.split(data,targets)
cv_matrix=[[tmp_index,test_index] for tmp_index,test_index in cv_generator]
for i,fold in enumerate(cv_matrix):
    #validate on the next fold test set (or on the first fold test set if last fold)
    #10 because 10 cross validation
    valid_index=cv_matrix[i+1][1] if i+1 < 10 else cv_matrix[0][2]
    #removes valid set from tmp_index
    train_index=[index for index in fold[0] if index not in valid_index]
    cv_matrix[i]=[train_index,valid_index,fold[1]]

verbose=False
fold_train_metrics,fold_valid_metrics,fold_test_metrics,fold_falses=[],[],[],[]
start = time()
np.random.seed(1)
save_to_print=""
fold=0
n_epochs=50#max n° of epochs the model will be trained to
patience = 10#n° of epochs without improvement during which the model will wait before stopping (if early_stopping)
run_CV=True#if False, breaks after one fold, else runs CV
early_stopping=True

for train_index,valid_index,test_index in cv_matrix:
    train_metrics,valid_metrics,test_metrics,falses=[],[],[],[]

    torch.manual_seed(1)#random seed for weights init
    model=Model(input_size, hidden_size,num_layers, bias,batch_first,
                 dropout,bidirectional, batch_size, output_size, is_lstm)
    if isinstance(model.encoder,torch.nn.LSTM):
        model.init_forget_bias()
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if fold==0:
        print("(total n° of parameters, n° of trainable parameters)\n",model.count_params())

    best_accuracy=0
    #best_loss=10000.0
    impatience=0
    for i in range(n_epochs):
        to_print=""
        np.random.shuffle(train_index)#shuffle training to facilitate SGD
        #training
        model.train()
        [loss,accuracy,sensitivity,specificity,ppv,npv],_=epoch(
        data, targets, model, optimizer, loss_fn, batch_size, train_index,clip,
            validation=False,window_size=window_size,task_i=task_i)
        train_metrics.append([loss,accuracy,sensitivity,specificity,ppv,npv])
        to_print+="\n\nfold n°{}, epoch n°{}, spent {}".format(fold,i,timeSince(start))
        to_print+="\nTRAINING : loss {:.3f}, accuracy {:.3f}".format(loss,accuracy)

        #validation
        model.eval()
        [loss,accuracy,sensitivity,specificity,ppv,npv],_=epoch(
            data, targets, model, optimizer, loss_fn, batch_size, valid_index,
            validation=True,window_size=window_size,task_i=task_i)
        valid_metrics.append([loss,accuracy,sensitivity,specificity,ppv,npv])
        to_print+="\nVALIDATION : loss {:.3f}, accuracy {:.3f}, sensitivity  {:.3f}, specificity {:.3f}, ppv {:.3f}, npv {:.3f}".format(
            loss,accuracy,sensitivity,specificity,ppv,npv)

        #patience update
        if accuracy <= best_accuracy:#`<=` no improvement is considered bad !#loss>=best_loss:
            impatience+=1
        else:
            best_accuracy=accuracy #best_loss=loss#
            impatience=0

        #test
        [loss,accuracy,sensitivity,specificity,ppv,npv],false=epoch(
            data, targets, model, optimizer, loss_fn, batch_size, test_index,
            validation=True,window_size=window_size,task_i=task_i)
        test_metrics.append([loss,accuracy,sensitivity,specificity,ppv,npv])
        falses.append(false)
        to_print+="\nTEST : loss {:.3f}, accuracy {:.3f}, sensitivity  {:.3f}, specificity {:.3f}, ppv {:.3f}, npv {:.3f}".format(
            loss,accuracy,sensitivity,specificity,ppv,npv)

        if verbose:
            print(to_print)
        save_to_print+=to_print

        #early stopping
        if impatience >= patience and early_stopping:
            save_to_print+="\nEarly stopped."
            break

    fold_train_metrics.append(train_metrics)
    fold_valid_metrics.append(valid_metrics)
    fold_test_metrics.append(test_metrics)
    fold_falses.append(falses)
    fold+=1
    if not run_CV:
        break
if not verbose:
    print("done")#save_to_print)

# Evaluation

if early_stopping:
    folds_lengths=[len(fold) for fold in fold_valid_metrics]
    print("number of epochs before early stopping for each fold:\n",folds_lengths)
    longest_fold=np.argmax(folds_lengths)
    shortest_fold=np.min(folds_lengths)
    average_stop=int(round(np.mean(folds_lengths)-patience))-1#-1 to index to it
    med_stop=int(round(np.median(folds_lengths)-patience-1))#-1 to index to it

try:
    if early_stopping:
        assert np.max([len(fold) for fold in fold_valid_metrics]) < n_epochs
        best_index=-patience-1
        stopped_at=[len(fold)-patience for fold in fold_valid_metrics]
    else:
        best_index=-1
        stopped_at=[n_epochs]
except AssertionError:
    print("The model didn't early stop therefore fold[-patience-1] is not the best epoch")
else:
    best_falses=[fold[best_index] for fold in fold_falses]
    flat_falses = sorted(flat_list(best_falses))
    results=return_results([fold[best_index] for fold in fold_train_metrics],
                  [fold[best_index] for fold in fold_valid_metrics],
                  [fold[best_index] for fold in fold_test_metrics],
                  stopped_at,
                   flat_falses)
    print(results)

# Save results
filename="experiments/"+str((task_i,is_lstm,learning_rate,hidden_size,num_layers,bidirectional,
dropout,clip,window_size))
file = open(filename+".csv","w")

file.write(results)
file.close()

# Save metrics
for metrics,metric_type in list(zip([fold_train_metrics,
fold_valid_metrics,
fold_test_metrics],["train","valid","test"])):
    with open(filename+metric_type, 'wb') as fp:
        pickle.dump(results, fp)
