Work by Paul Lerner under Laurence Likforman-Sulem supervision.

All the modules, models, extra-code are stored in `modules/`

All of the content of this Readme is also present in relevant places in the different python-notebook files.

If you use this software, please cite our paper : *Lerner, P., Likforman-Sulem, L., 2019. Classification of online handwriting time series for parkinson’s disease diagnosis using deep learning, in: 2019 Junior Conference on Data Science and Engineering (JDSE).*
```
@inproceedings{lerner2019jdse,
  title={Classification of online handwriting time series for Parkinson's Disease diagnosis using Deep Learning},
  author={Paul Lerner and Laurence Likforman-Sulem},
  booktitle={2019 Junior Conference on Data Science and Engineering (JDSE)},
  year={2019}
}
```
# Contents
- [Dependencies](#dependencies)
- [Utils](#utils)
- [Main](#main)
  * [utils](#utils)
  * [Data](#data)
    + [Downloading](#Downloading)
    + [Loading](#loading)
    + [Massaging](#massaging)
    + [Some insights about the number of timesteps in the different tasks](#some-insights-about-the-number-of-timesteps-in-the-different-tasks)
  * [Training](#training)
    + [Cross-Validation (CV)](#cross-validation--cv-)
    + [Majority Voting](#majority-voting)
    + [Epoch](#epoch)
    + [Step](#step)
    + [Hyperparameters](#hyperparameters)
      - [Random Search](#random-search)
  * [Visualization](#visualization)
    + [Metrics](#metrics)
    + [Interpretation](#interpretation)
- [NewHandPD](#NewHandPD)
- [Data Exploration](#Data-Exploration)

<!---# Main
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
`python main.py f 1e-3 10 4 f 0.0 5.0 0 none`  
If you want to change any other hyperparameters you will have to dive into the `main.py` code.
--->
# Dependencies
- PyTorch
- Sci-kit learn
- NumPy (should be included in PyTorch)
- Seaborn (very optional, only used to set the style of matplotlib)
- SciPy (only for [Data Exploration](#Data-Exploration) and [NewHandPD](#NewHandPD))

# Utils
`utils.py` contains misc. utilitaries such as measures and task dictionaries, confusion matrix.
# Main

## utils
some extra utils in addition to `utils.py` due to the great number of parameters :
- plot plots either the loss or the accuracy of the model depending on `plot_i` (see `plot2index`)
- return_results formats the hyperparameters, the evaluation metrics and the index of the wrongly classified subjects to a nice string, separated by `;`
- print_results prints the above string for the epoch which provided the best accuracy in average among all 10 folds.

## Data
### Downloading

To get access to the PaHaW database
Please	 fill	 in	 a	 license	 agreement	 that	 can	 be	 downloaded	 in	 DOCX	 or	 PDF	 (see http://bdalab.utko.feec.vutbr.cz/)	 and	 send	 it	 to	 mekyska@feec.vutbr.cz	 with	 CC	 to	peter.drotar@tuke.sk	 and	 irena.rektorova@fnusa.cz.	 You	 will	 consequently	 get	 an	 access	 to	 the	database.

The data should be placed like this :  
```
data
└───PaHaW
│   └───PaHaW_public
│   │   │   000**
│   │   corpus_PaHaW.xlsx
│   │   corpus_PaHaW.csv
```

**`corpus_PaHaW.csv` doesn't exist in the database, you should create it by opening `corpus_PaHaW.xlsx` (with e.g. LibreOffice Calc or Microsoft Excel) then `File, Save as, corpus_PaHaW.csv` (the separator should be `;`).**
### Loading
is done in `load_data.py`. The load_data function yields subjects data, label (1 for PD 0 for control) and age. Note that it discards the subjects who didn't perform the spiral, i.e. Subjects 46 (control), 60 (PD) and 66 (control), counting from zero. Moreover, it trims the data for the few subjects who begin their exam in-air and the one where there is a measure error at the end, see [Data Exploration](#Data-Exploration). The raw data is then turned into a list in `Main`. It will be saved using pickle in `join("data","raw_data")` (i.e. `data/raw_data` in UNIX and Linux). In the future, raw data will be loaded directly from this path.

### Massaging
You can then select a task using `task_i` (cf. task2index). This is done in task_selection which is called in massage_data. Set `task_i` to None if you want to keep the whole eight tasks in the shape of (72 subjects, 8 task, x timesteps, 7 measures). Unfortunately, you have to select a task if you want to split it into strokes using `paper_air_split`. This is done in massage_data. After eventually splitting data into strokes, massage_data standardizes the data (giving it a mean of 0 and a std of 1) :
- t0 is removed from each timestamps so the time stamp measure represents the length of the exams
- Moreover, the timestamp is standardized globally, i.e. we compute the mean and std value of all subjects in a given task before centering and reducing the data
- unlike the timestamp, the other measures are scaled subject-wise, independently of all subjects, using [sklearn's scale](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html)

After this, massage_data zero-pads (and trims if `trim`) the data to `max_len` if `max_len is not None`. This can also be done in epoch before feeding the subject to the model, see [Training](#Training).
The massaged data is then saved to a numpy array in `join("data","<custom name>")`. In the future, data will be loaded directly from this path.

### Some insights about the number of timesteps in the different tasks

These are stored as lists in `utils.py`

 The task sequence is in average **2286** &rarr; task duration is in average 11.4s  

 task | max duration | avg duration | avg duration std | avg duration per letters | max duration per tokens | avg duration per stroke | max duration per stroke | max # of strokes | std of stroke duration | std of # of strokes
 --|--|--|--|--|--|--|--|--|--|--
 spiral | 16071 | 2873 | 2267.76 | 2758.75  | 16071| 1189 +- 1715 | 16071|25|1715|3.57
 l       | 4226|1639 | 724.14 | 333 | 1242  |169 +- 109|752   | 15|109 | 1.42
 le       | 6615| 1966 | 999.61 | 198 | 1649|202 +- 160|1104 | 15| 160 | 1.57
 les     | 6827| 2301 | 1095.54 | 153| 1956|228 +- 183|1476 | 21| 183 | 2.47
 lektorka | 7993| 2600 | 1333.76  | 163| ?|203 +- 281|3568 | 29| 281 | 7.16
 porovnat | 5783| 2314 | 1033.69 |144| ?|148 +- 198|2057 | 43| 198 | 8.72
 nepopadnout | 4423| 1473 | 650.65 | 133| ?|134 +- 189|2267 | 35| 189 | 6.85
 tram       | 7676| 3094 | 1158.81 | 146| ?|101 +- 117|1231 | 67| 117 | 8.38

## Training
The Cross-Validation (CV) function and the Majority Voting function are implemented in Main because of the great number of parameters. The training step and epoch function are in `training.py`

### Cross-Validation (CV)

You can use the CV function even if you want to train on only one fold (i.e. one dataset split) by setting `run_CV` to `False`. In this case, you can use `start_at_fold` to skip the the first `start_at_fold` folds (thus it should be smaller than `n_splits`, but that's up to the user).

You can choose the number of splits (e.g. 10 fold cross validation) using n_splits and the # of epochs using n_epochs. We only used 10 CV in all our experiments to keep consistent results with Drotar et al. and Moetesum et al. The actual split is done using [sklearn's StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), with a random_state set at `1` so that the split is always the same. The random seed is always set at `1` in NumPy and PyTorch before doing any random operation. Note that the CV is stratified, meaning that there's the same # of PDs and HCs in the data sets, see table below.
##### Stratified 10 CV Dataset Split
fold | subject (index, label)
--|--
0 | (3, 1), (25, 0), (41, 1), (43, 1), (58, 1), (63, 0), (65, 0), (69, 0)
1 | (17, 1), (19, 1), (21, 1), (30, 1), (48, 0), (50, 0), (52, 0), (56, 0)
2 | (36, 1), (37, 1), (42, 1), (57, 1), (61, 0), (62, 0), (64, 0), (68, 0)
3 | (2, 1), (4, 1), (24, 0), (26, 0), (31, 1), (35, 1), (59, 0), (60, 0)
4 | (10, 1), (14, 1), (29, 1), (38, 0), (45, 0), (53, 0), (55, 1), (67, 0)
5 | (6, 1), (18, 1), (20, 1), (28, 0), (49, 0), (51, 0), (54, 1), (66, 0)
6 | (7, 1), (13, 1), (32, 0), (44, 0), (70, 0), (71, 1)
7 | (0, 1), (1, 1), (16, 1), (22, 0), (23, 0), (47, 0)
8 | (5, 1), (11, 1), (15, 1), (27, 0), (39, 0), (46, 0)
9 | (8, 1), (9, 1), (12, 1), (33, 0), (34, 0), (40, 0)

If you want to use 2 different models for on-paper and in-air strokes, you should dive into the CV code and initialize `in_air` the same way as `model`.

The model is trained using the [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) with the learning rate defined in [Hyperparameters](#Hyperparameters), the other hyperparameters are kept to the PyTorch's default setting (which respects the advices of the original paper).

At the end of the Cross-validation, if `run_CV`, the program will dump the `fold_train_metrics` and the `fold_test_metrics` using Pickle in `join("experiments",model_name)+<train or test>`, with `model_name` being the concatenation of `model_type` and other hyperparameters. In the same way, it will save the models' predictions in `join("predictions",model_name)+<train or test>`. It will also save the results (as defined in [utils](#utils)) to the gitignored-file : `results.csv`.

### Majority Voting

The majority voting function allows you to combine the predictions of the different model, either directly stored in `model_test_predictions` or loaded from the `predictions` folder.

You can control whether to perform actual majority voting or to fuse the models' predictions' probability using `round_before_voting`. The rest of the function is very similar to CV.

### Epoch
`batch_size` should be `1`, you're welcome to implement batch learning but it's not suited for a small dataset like PaHaW.

Depending on `paper_air_split` and `hierarchical` (see [Hyperparameters](#Hyperparameters)), the epoch function will either :
- go through all the data which index is in `random_index` (provided by the CV function), or
- if `paper_air_split and not hierarchical`, the model will shuffle the strokes of every subjects, go through every of them, saving the model predictions in the `predictions` *dictionary*, then fuse the predictions of the model for each subject (i.e. task). If you want to use majority voting rather than prediction's provability averaging, you will have to dive in the code under the line : `predictions=[np.mean(sub) for sub in list(predictions.values())]`
- **todo : remove augmentation**

Regardless of `paper_air_split` and `hierarchical` the epoch function will zero-pad (but not trim) the data to `max_len` if not `None`.

### Step
The step functions either performs a training step using the above model, optimizer and loss function (see [Hyperparameters](#Hyperparameters)) or a simple forward pass using the model if `validation`. In the former case, the step function may clip the gradients norm of the RNN (part of the) model before back-propagating, if you use a RNN and if `clip is not None`.

### Hyperparameters
The loss function should always be [Binary Cross Entropy](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss) as all our models have a Sigmoid activation and are trained to a binary classification task.

You can choose which model to use with `model_type` by selecting one of :
- lstm
- gru
- hrnn
- tcn (from [Locus Lab](https://github.com/locuslab/TCN))
- pretrained (see below)
- cnn1d
- hcnn1d
- hscnn1d
- hsclstm
- hscgru


They all are defined in `modules/`. If you select `pretrained` you should also define `pre_trained_name` which should correspond to the name of a pre-trained model stored in `weights/`.

Note that the hyperparameters are not always relevant for the chosen `model_type`. In this case they won't matter. Give them a random value.

`input_size` refers to the number of measures/features that you are using to train the model (default : 7).

The CorrectHyperparameters function is defined in `utils.py` and makes the CNNs hyperparameters consistent with the input shape, e.g. the pooling kernel should divide the input length.

All the other hyperparameters are pretty straightforward so I'll let you refer to the PyTorch documentation :
- [LSTM](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM)
- [GRU](https://pytorch.org/docs/stable/nn.html#torch.nn.GRU)
- [1D Convolution](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d)
- [Linear](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear)

#### Random Search
You can perform a random search of the hyperparameters by setting `random_search` to True. You can play with the different ranges of the hyperparameters using the `ranges` dictionary.

## Visualization

### Metrics

This allows you to visualize the results of your model using the plot function defined in `utils`.

You can either visualize the results currently stored in `fold_test_metrics` or load them from the `experiments` folder.

###  Interpretation

This allows you to visualize the weights of the model, However this is an ad-hoc example for `cnn1d`

## Results
All of our experiment results, including 100s of random-search experiments are available in `reports/experiment results.ods` (careful of the space).

The results' format is described in [utils](#utils).
# NewHandPD
## Data
### Download
#### Command-line
To get the data you can run the following commands from the `data` directory :
```
mkdir NewHandPD
cd NewHandPD/

#download healthy data
mkdir Healthy
cd Healthy/
wget http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewHealthy/Signal.zip
unzip Signal.zip
cd ..

#download Patients data
mkdir Patients
cd Patients/
wget http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewPatients/Signal.zip
unzip Signal.zip
```
#### Graphic interface
You can get the data from [here](http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/), under "The NewHandPD dataset" you will find :
- [Healthy Signals](http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewHealthy/Signal.zip)
- [Patient Signals](http://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewPatients/Signal.zip)

The data should be placed like this :  
```
data
└───NewHandPD
│   └───Healthy
│   │   └───Signal
│   │   │   │   *.txt
│   └───Patients
│   │   └───Signal
│   │   │   │   *.txt
```

### Correct it /!\ Mandatory /!\
Healthy's `SigMea1-28` is not properly named it should be `SigMea1-H28`, rename it before running the rest of this code !

Healthy `sigDiaA-H34.txt` age was set at 50 but in all other tasks it was set at 40 so I set it to 40, do it as well before running the rest of the code ! See :  
circA-H34.txt 40 circB-H34.txt 40 sigDiaB-H34.txt 40 sigMea1-H34.txt 40 sigMea2-H34.txt 40 sigMea3-H34.txt 40 sigMea4-H34.txt 40 sigSp1-H34.txt 40 sigSp2-H34.txt 40 sigSp3-H34.txt 40 sigSp4-H34.txt 40

## Data analysis
The first part consists of a statistical analysis :
- correlation between the subject's age and it's different tasks' duration
- t-tests between the different tasks durations of controls and PDs
- t-test between the controls and PDs' age

The second part consists in classification of subjects depending on the duration of their different task, and of their age (using either a rule based model or Linear discriminant analysis).

# Data exploration
## Age and duration analysis
this study is similar to the one in [NewHandPD](#NewHandPD).
The first part consists in classification of subjects depending on the duration of their different task, and of their age (using either a rule based model or Linear discriminant analysis).

The second part consists of a statistical analysis :
- correlation between the subject's age and it's different tasks' duration
- t-tests between the different tasks durations of controls and PDs
- t-test between the controls and PDs' age

## Exploration

Provides some insights about the content of the data

## Recording errors
These recording errors are now corrected in the load_data function as explained in [Loading](#Loading)

All the subject’s numbers I mention are counting from
zero and after excluding the subjects that didn’t perform the
spiral exam.
### Exam starting in air

The exams spiral, l, lektorka and tram from subjects 56,
9, 39 and 67, respectively, started their exam while pen was
in air (i.e. not on paper). Although Drotar et al. say that the recording
starts when the pen first touches the paper so I assume it’s a
recording error and discarded the in-air points.
### Exam finishing in air
In the same way the subject # 43 has a recording problem
on his tram exam : the timestamp measure jumps to 10<sup>12</sup> on
the 12 last points of the recording although the timestamps
from all subjects are in the 10<sup>6</sup> magnitude. So I assume it’s
a recording error and discarded the 12 last points.
