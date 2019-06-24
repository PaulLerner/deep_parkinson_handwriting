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
last_stroke_in_air_index=[[],#spiral
    [4, 36, 71],#l
    [11, 14, 16, 42],#le
    [1, 13, 14, 20, 54]#les
]
non_letters_indexes=[[],#spiral
    [(22,1), (26,2), (36,5), (37,1), (41,4), (46,4), (48,1),(3,4),
    (3,2),(6,5), (6,3),  (14,6), (14,4),(14,2), (16,6), (16,4), (16,2), (21,5), (71,6), (71,2)],#l

    [(3,4), (6,5), (6,4), (6,2), (9,4), (9,3), (11,5), (12,1), (13, 1),
     (14, 6), (14, 1), (16, 5), (18, 3), (18, 2), (18, 1), (20, 3), (26, 2),
     (26, 1), (27, 4), (41, 5), (41, 2), (42, 7), (42, 5), (42, 3), (65, 5), (65, 3)],#le

      [(1, 7),(1, 6),(3, 4),(6, 4),(6, 1),(9, 1),(13, 5),(14, 10), (14, 9), (14, 8), (14, 7),(14, 4),(14, 2),
       (18, 4), (18, 3), (18, 2), (18, 1),(20, 8),(20, 6),(20, 4),(20, 2),(23, 4),(26, 4),(26, 1),(38, 3),
      (48, 4),(50, 4),(54, 9),(54, 7),(54, 5),(54, 3),(54, 1),(62, 4),(65, 6),(65, 4),(65, 1)]#les
]
too_many_letters_indexes=[[],#spiral
    [12, 21, 23, 44, 67],#l
    [],#le
    [1,37,62]#les
]
def LetterSplit(data,task_i):
    print("Merging strokes into letters")
    for j in range(len(data)):
        tmp=[]
        for i in range(0,len(data[j]),2):
            try :
                data[j][i+1]
            except IndexError:
                tmp.append(data[j][i])
            else:
                tmp.append(np.concatenate((data[j][i],data[j][i+1]),axis=0))
        data[j]=tmp
    def pop(i,j):
        data[i][j-1]=np.concatenate((data[i][j-1],data[i][j]))
        data[i].pop(j)
    for i,j in non_letters_indexes[task_i]:
        pop(i,j)
    for i in too_many_letters_indexes[task_i]:
        data[i].pop()
    assert [i for i,s in enumerate(data) if len(s) != 5]==[]
    return data

def DiscardNonLetters(data,task_i):
    print("discarding non letters from stroke list")
    for i,j in non_letters_indexes[task_i]:
        if 2*j+1<len(data[i]):
            data[i].pop(2*j+1)
        data[i].pop(2*j)
    for i in too_many_letters_indexes[task_i]:#did 6 l instead of 5
        data[i].pop()
        data[i].pop()
    for i in last_stroke_in_air_index[task_i]:#in air stroke after last l
        data[i].pop()
    assert [i for i,s in enumerate(data) if len(s) != 9]==[]
    return data
def massage_data(data,task_i,compute_speed_accel_,compute_movement_,downsampling_factor,
window_size,paper_air_split=False,newhandpd=False,max_len=None,letter_split=False,discard_non_letters=False,pad_subs=False,trim=False):
    """
    returns data
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

    ## Split in subsequence (or not)
    #Set `window_size` to `None` if you don't want to split data into subsequence of fixed length
    if task_i is not None:
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
                data[j]=task
            if letter_split:#todo : rename in token split
                data=LetterSplit(data,task_i)
            elif discard_non_letters:
                data=DiscardNonLetters(data,task_i)
            print("len(data), data[0].shape, total n° of subsequences (i.e. training examples) :")
            print(len(data),",",len(data[0]),len(data[0][0]),len(data[0][0][0]),",",sum([len(subs) for subs in data]))
        else:
            print("the task is represented as one single sequence  (i.e. data was not transformed)")

    if window_size is not None or paper_air_split or task_i is None:#subsequences or multiple tasks

        print('computing global means')
        for i,subject in enumerate(data):
            for j,sub in enumerate(subject):
                #removes t0 from each timestamps so the time stamp measure represents the length of the exams
                data[i][j][:,measure2index["timestamp"]]-=data[i][j][0,measure2index["timestamp"]]
        if task_i is None:
            #computes overall measures and stds per task
            data=np.asarray(data)
            means,stds=[],[]
            for task in range(data.shape[1]):
                flat=flat_list(data[:,task])
                means.append(np.mean(flat,axis=0)[measure2index["timestamp"]])
                stds.append(np.std(flat,axis=0)[measure2index["timestamp"]])
        else:
            #computes overall measures and stds
            flat=np.asarray(flat_list(flat_list(data)))
            means,stds=np.mean(flat,axis=0)[measure2index["timestamp"]],np.std(flat,axis=0)[measure2index["timestamp"]]
        print("scaling")
        for i,subject in enumerate(data):
            for j,sub in enumerate(subject):
                data[i][j]=scale(sub,axis=0)
                #keep the button_status unscaled
                data[i][j][:,[measure2index["button_status"]]]=sub[:,[measure2index["button_status"]]]
                #globally scale the timestamp
                if task_i is None:
                    data[i][j][:,[measure2index["timestamp"]]]=(sub[:,[measure2index["timestamp"]]]-means[j])/stds[j]
                else:
                    data[i][j][:,[measure2index["timestamp"]]]=(sub[:,[measure2index["timestamp"]]]-means)/stds
                if downsampling_factor != 1:
                    if i ==0 and j==0:
                        print("and downsampling")
                    data[i][j]=decimate(data[i][j], downsampling_factor,axis=0)#then downsample
                    #rounds the button status because decimate applies a filter
                    data[i][j][:,[measure2index["button_status"]]]=[[round(b[0])] for b in data[i][j][:,[measure2index["button_status"]]]]
    else:
        print('computing global means')
        for i in range(len(data)):
            #removes t0 from each timestamps so the time stamp measure represents the length of the exams
            data[i][:,measure2index["timestamp"]]-=data[i][0,measure2index["timestamp"]]
        #computes overall measures and stds
        flat=np.asarray(flat_list(data))
        means,stds=np.mean(flat,axis=0)[measure2index["timestamp"]],np.std(flat,axis=0)[measure2index["timestamp"]]
        ## Scale then downsample (or not) then concatenate task id (or not)
        print("scaling")
        for i,subject in enumerate(data):
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
    if max_len is not None:
        print("padding data at {} timesteps. Trimming : {} ".format(max_len,trim))
        if task_i is None :
            for i,subject in enumerate(data):
                for j,task in enumerate(subject):#task
                    if len(task) > max_len[j]:
                        if trim:
                            data[i][j]=task[:max_len[j]]
                    else:
                        data[i][j]=np.concatenate((task,np.zeros(shape=(max_len[j]-len(task),task.shape[1]))))
        elif window_size is not None or paper_air_split :
            for i,subject in enumerate(data):
                for j,sub in enumerate(subject):#sub
                    if len(sub) > max_len:
                        if trim:
                            data[i][j]=sub[:max_len]
                    else:
                        data[i][j]=np.concatenate((sub,np.zeros(shape=(max_len-len(sub),sub.shape[1]))))
                if pad_subs:
                    if i == 0:
                        print("padding # of subsequences to",max_strokes[task_i])
                    for _ in range(max_strokes[task_i]-len(subject)):
                        data[i].append(np.zeros(shape=(max_len,sub.shape[1])))
        else:#only one task
            for i,task in enumerate(data):
                if len(task) > max_len:
                    if trim:
                        data[i]=task[:max_len]
                else:
                    data[i]=np.concatenate((task,np.zeros(shape=(max_len-len(task),task.shape[1]))))
    print("converting data to numpy array")
    data=np.asarray(data)
    print("data shape :",data.shape)
    return data
