import numpy as np
from os.path import join
from os import listdir
from utils import *
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample
from scipy.signal import decimate

def load_data():
    data_path=join("..","PaHaW","PaHaW_public")#/00026/00026__1_1.svc"
    folder_path=listdir(data_path)
    folder_path.sort()

    meta_path=join("..","PaHaW","corpus_PaHaW.csv")
    meta_data=np.loadtxt(meta_path,dtype=str,skiprows=1,delimiter=";")#skip the first line == headers
    labels=list(map(lambda x: 1 if x =="ON" else 0, meta_data[:,4]))

    #Subjects 46 (control), 60 (PD) and 66 (control) didn't perform the spiral !

    #data=[]
    for i,folder in enumerate(folder_path):
        subject=[]
        task_path=listdir(join(data_path,folder))
        task_path.sort()
        if len(task_path)!=8:#subject didn't perform the spiral
            #so we discard it
            continue
            #subject.append([])#add an empty array so that all tasks are on the same column number
        for task in task_path:
            path=join(data_path,folder,task)
            #load data as float (not int because we will need to standardize it afterwards)
            #and throw out the first line == number of lines in the file
            subject.append(np.loadtxt(path, dtype=float, skiprows=1,delimiter=" "))
        yield subject,labels[i]
        #data.append(subject)

    #discard the subjects that didn't perform spiral
    #targets= [labels[i]  for i,subject in enumerate(data) if len(subject[0])!=0]
    #data=[subject for subject in data if len(subject[0])!=0]
    #return data, targets

def massage_data(task_i,compute_movement,downsampling_factor,window_size):
    ## Loading
    #Cf `load_data.py`
    data_gen=load_data()
    data,targets=[]
    for subject,label in data_gen:
        data.append(subject)
        targets.append(label)
    print("(75-3 subjects, 8 tasks, X timesteps, 7 measures)")
    print(len(data),len(data[0]),len(data[0][0]),len(data[0][0][0]))

    ## Task selection
    #set `task_i` to None if you want to train the model on all tasks at once (i.e. early fusion)
    #Else set `task_i` to the desired task index (cf. task2index)

    #task_i=0
    if task_i is not None:
        print("\ntask index, name")
        print(task_i,index2task[task_i])
        #keep only one task
        data=[subject[task_i] for subject in data]
        #keep only one measure
        #data=[[[raw[i][task][j][6]] for j in range(len(raw[i][task])) ]  for i,subject in enumerate(raw) if len(raw[i][task])!=0]#discard the subjects that didn't perform spiral
    else:
        print("task_i is None so we will use all tasks to train the model")
    print("len(data), len(targets), len(data[0]) :")
    print(len(data),len(targets),len(data[0]))

    ## Compute movement
    #Transforms data as Zhang et al. (cf Report #5)

    #compute_movement=False
    if compute_movement:
        print("computing movement\n")
        button_i=measure2index["button_status"]
        for i,task in enumerate(data):
            for t in range(len(task)-1):
                button=task[t+1][button_i]*task[t][button_i]
                data[i][t]=task[t+1]-task[t]
                data[i][t][button_i]=button
            data[i]=data[i][:-1]#throw out the last point
    else:
        print("\nmovement was not computed (i.e. data was not transformed)\n")

    ## Scale then downsample (or not) then concatenate task id (or not)

    #downsampling_factor=1
    for i,subject in enumerate(data):
        if task_i is not None:
            if downsampling_factor==1:#don't downsample
                if i ==0:
                    print("scaling")
                data[i]=scale(subject,axis=0)
            else:
                if i ==0:
                    print("scaling and downsampling")
                data[i]=decimate(
                    scale(subject,axis=0),#scale first
                          downsampling_factor,axis=0)#then downsample
        else:
            for j, task in enumerate(subject):
                if downsampling_factor==1:#don't downsample
                    if i ==0 and j ==0:
                        print("scaling and concatenating task id")
                    #concatenate task id and actual measures
                    data[i][j]=np.concatenate(
                        ([one_hot[j] for _ in range(len(task))],#create a matrix of the same shape as the task
                         scale(task,axis=0)),#scales the task
                        axis=1)#then concatenate the two
                else:
                    raise NotImplementedError("downsampling is not implemented for Multi-task learning")
    print("len(data), len(targets), len(data[0]) :")
    print(len(data),len(targets),len(data[0]))

    ## Split in subsequence (or not)
    #Set `window_size` to `None` if you don't want to split data into subsequence of fixed length

    #window_size=None#Set to None if you don't want to split data into subsequence of fixed length
    overlap=90
    if window_size is not None:
        print("\nsplicing data into subsequences")
        for i,task in enumerate(data):
            data[i]=[task[w:w+window_size] for w in range(0,len(task)-window_size,window_size-overlap)]
        print("len(data), data[0].shape, total n° of subsequences (i.e. training examples) :")
        print(len(data),",",len(data[0]),len(data[0][0]),len(data[0][0][0]),",",sum([len(subs) for subs in data]))
    else:
        print("the task is represented as one single sequence  (i.e. data was not transformed)")

    return data, targets
