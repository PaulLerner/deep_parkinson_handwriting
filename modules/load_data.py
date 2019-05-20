import numpy as np
from os.path import join
from os import listdir
from modules.utils import *
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample
from scipy.signal import decimate
import warnings

def load_data():
    """
    data generator : yields subject data, label and age subject by subject
    trims data with error measure (cf. data exploration)"""
    data_path=join("..","PaHaW","PaHaW_public")#/00026/00026__1_1.svc"
    folder_path=listdir(data_path)
    folder_path.sort()

    meta_path=join("..","PaHaW","corpus_PaHaW.csv")
    meta_data=np.loadtxt(meta_path,dtype=str,skiprows=1,delimiter=";")#skip the first line == headers
    labels=list(map(lambda x: 1 if x =="ON" else 0, meta_data[:,4]))
    ages=meta_data[:,5].astype(int)
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
        for task_name in task_path:
            path=join(data_path,folder,task_name)
            #load data as float (not int because we will need to standardize it afterwards)
            #and throw out the first line == number of lines in the file
            task=np.loadtxt(path, dtype=float, skiprows=1,delimiter=" ")
            if task[0][measure2index["button_status"]]!=1:#exam starts in air
                for k,timestep in enumerate(task):
                    if(timestep[measure2index["button_status"]]==1):#wait for on paper button status
                        break
                #then trims the data
                task=task[k:]
            elif any(task[:,measure2index["timestamp"]]>1e7):#defect of recording (see data exploration)
                task=task[:-12]
            subject.append(task)
        yield subject,labels[i],ages[i]

        """for i,j in [(56, 0), (9, 1), (39, 4), (67, 7)]:
            for k,timestep in enumerate(data[i][j]):
                if(timestep[measure2index["button_status"]]==1):#wait for on paper button status
                    break
            #then trims the data
            data[i][j]=data[i][j][k:]
        data[43][-1]=data[43][-1][:-12]"""

## augmentation
def flip(task,axis_i):
    if axis_i is not 0 and axis_i is not 1:
        raise ValueError("expected 0 or 1 for value of axis_i, got {}".format(axis_i))
    axis=task[0][axis_i]
    for i,point in enumerate(task[:,axis_i]):
        if point < axis:
            task[i][axis_i]=axis+(axis-point)
        else:
            task[i][axis_i]=axis-(point-axis)
    return task
def rotate(task, delta_rotate):
    x0=task[0][0]#angle starts here
    y0=task[0][1]
    for i, (y,x) in enumerate(task[:,:2]):
        vector=[x-x0,y-y0]
        norm=np.linalg.norm(vector)
        angle=np.angle(vector[0]+vector[1]*1j)#*1j to add imaginary part to y-coordinate
        task[i][1]=np.cos(angle+delta_rotate)*norm#new x
        task[i][0]=np.sin(angle+delta_rotate)*norm#new y
    return scale(task,axis=0)#recenters the task

#rotated=rotate_(task.copy(),np.pi/10)
"""
h_flip=horizontal_flip(task.copy())
v_flip=vertical_flip(task.copy())
double_flip=horizontal_flip(v_flip.copy())
translation=np.random.rand()-0.5#because the std is one
translated=task.copy()
translated[:,0]+=translation
translated[:,1]+=translation
#~ match the translation scale
#as the standardized data ranges ~ from -2 to 2
zoom_factor=np.random.uniform(0.8,1.2)
zoomed=task.copy()
zoomed[:,0]*=zoom_factor
zoomed[:,1]*=zoom_factor"""

## preprocessing
def compute_movement(data):
    """Compute movement
    Transforms data as Zhang et al. (cf Report #5)"""
    print("computing movement\n")
    button_i=measure2index["button_status"]
    for i,task in enumerate(data):
        for t in range(len(task)-1):
            button=task[t+1][button_i]*task[t][button_i]
            data[i][t]=task[t+1]-task[t]
            data[i][t][button_i]=button
        data[i]=data[i][:-1]#throw out the last point
    return data

def task_selection(data,task_i,newhandpd=False):
    """set `task_i` to None if you want to train the model on all tasks at once (i.e. early fusion)
    Else set `task_i` to the desired task index (cf. task2index)
    """
    if task_i is not None:
        print("\ntask index, name")
        print(task_i,index2task[task_i])
        #keep only one task
        data=[subject[task_i] for subject in data]
        #keep only one measure
        #data=[[[raw[i][task][j][6]] for j in range(len(raw[i][task])) ]  for i,subject in enumerate(raw) if len(raw[i][task])!=0]#discard the subjects that didn't perform spiral
    elif newhandpd:
        print("setting task_i to -1")
        task_i=-1
    else:
        print("task_i is None so we will use all tasks to train the model")
    print("len(data), len(data[0]) :")
    print(len(data),len(data[0]))
    return data
def compute_speed_accel(data):
    """on single task training, concatenates the instantaneous speed and acceleration to each timestep of the data.
    Thus the data is 2 timesteps shorter (we discard the first 2)"""
    print("computing speed and acceleration")
    for i,task in enumerate(data):
        speed=np.zeros((len(task)-1,1))
        for t in range(len(task)-1):
            speed[t][0]=np.linalg.norm(#norm of vector
                    task[t+1][:2]-task[t][:2]#vector [y(t+1)-y(t) , x(t+1)-x(t)]
                )
        accel=np.zeros((len(speed)-1,1))
        for t in range(len(speed)-1):
            accel[t][0]=speed[t+1]-speed[t]

        #discard the 1st speed point
        speed_accel=np.concatenate((speed[1:],accel),axis=1)
        #discard the 2 firsts timesteps
        data[i]=np.concatenate((task[2:],speed_accel),axis=1)
    return data

def massage_data(data,targets,task_i,compute_speed_accel_,compute_movement_,downsampling_factor,window_size,paper_air_split=False,newhandpd=False):
    """
    returns data, targets
    set `task_i` to None if you want to train the model on all tasks at once (i.e. early fusion)
    Else set `task_i` to the desired task index (cf. task2index)
    compute_movement Transforms data as Zhang et al. (cf Report #5)
    Set `downsampling_factor` to `1` if you don't want to downsample
    Set `window_size` to `None` if you don't want to split data into subsequence of fixed length
    Set `paper_air_split` to `False` if you don't want to split data into strokes
    """
    data=task_selection(data,task_i,newhandpd)
    if compute_speed_accel_:
        data=compute_speed_accel(data)
    elif compute_movement_:
        data=compute_movement(data)
    else:
        print("\nneither speed nor movement was computed (i.e. data was not transformed)\n")

    for i in range(len(data)):
        #removes t0 from each timestamps so the time stamp measure represents the length of the exams
        data[i][:,measure2index["timestamp"]]-=data[i][0,measure2index["timestamp"]]
    if task_i is not None:
        #computes overall measures and stds
        flat=np.asarray(flat_list(data))
        means,stds=np.mean(flat,axis=0)[measure2index["timestamp"]],np.std(flat,axis=0)[measure2index["timestamp"]]
    ## Scale then downsample (or not) then concatenate task id (or not)
    for i,subject in enumerate(data):
        if task_i is not None:
            if i ==0:
                print("scaling",end=" ")
            data[i]=scale(subject,axis=0)
            #keep the button_status unscaled
            data[i][:,[measure2index["button_status"]]]=subject[:,[measure2index["button_status"]]]
            #globally scale the timestamp
            data[i][:,[measure2index["timestamp"]]]=(subject[:,[measure2index["timestamp"]]]-means)/stds
            if downsampling_factor != 1:
                if i ==0:
                    print("and downsampling")
                data[i]=decimate(data[i], downsampling_factor,axis=0)#then downsample
                #rounds the button status because decimate applies a filter
                data[i][:,[measure2index["button_status"]]]=[[round(b[0])] for b in data[i][:,[measure2index["button_status"]]]]
        else:
            warnings.warn("you're standardizing the button_status",Warning)
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
    print("\nlen(data), len(targets), len(data[0]) :")
    print(len(data),len(targets),len(data[0]))

    ## Split in subsequence (or not)
    #Set `window_size` to `None` if you don't want to split data into subsequence of fixed length
    overlap=90
    if window_size is not None:
        print("\nsplitting data into subsequences")
        for i,task in enumerate(data):
            data[i]=[task[w:w+window_size] for w in range(0,len(task)-window_size,window_size-overlap)]
        print("len(data), data[0].shape, total n° of subsequences (i.e. training examples) :")
        print(len(data),",",len(data[0]),len(data[0][0]),len(data[0][0][0]),",",sum([len(subs) for subs in data]))
    elif paper_air_split:
        print("\nsplitting data into strokes")
        for j, task in enumerate(data):
            changes = []
            for i in range(len(task)-1):
                if task[i][measure2index["button_status"]]!=task[i+1][measure2index["button_status"]]:
                    changes.append(i+1)
            task=np.split(task,changes)
            if task[0][0][measure2index["button_status"]]!=on_paper_value:
                task.pop(0)
            data[j]=task
        print("len(data), data[0].shape, total n° of subsequences (i.e. training examples) :")
        print(len(data),",",len(data[0]),len(data[0][0]),len(data[0][0][0]),",",sum([len(subs) for subs in data]))
    else:
        print("the task is represented as one single sequence  (i.e. data was not transformed)")

    return data, targets
