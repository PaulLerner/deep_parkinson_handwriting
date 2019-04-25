import numpy as np
from os.path import join
from os import listdir

def load_data():
    data_path=join("..","PaHaW","PaHaW_public")#/00026/00026__1_1.svc"
    folder_path=listdir(data_path)
    folder_path.sort()

    meta_path=join("..","PaHaW","corpus_PaHaW.csv")
    meta_data=np.loadtxt(meta_path,dtype=str,skiprows=1,delimiter=";")#skip the first line == headers
    labels=list(map(lambda x: 1 if x =="ON" else 0, meta_data[:,4]))

    #Subjects 46 (control), 60 (PD) and 66 (control) didn't perform the spiral !

    data=[]
    for folder in folder_path:
        subject=[]
        task_path=listdir(join(data_path,folder))
        task_path.sort()
        if len(task_path)!=8:#subject didn't perform the spiral
            subject.append([])#add an empty array so that all tasks are on the same column number
        for task in task_path:
            path=join(data_path,folder,task)
            #load data as float (not int because we will need to standardize it afterwards)
            #and throw out the first line == number of lines in the file
            subject.append(np.loadtxt(path, dtype=float, skiprows=1,delimiter=" "))
        data.append(subject)

    #discard the subjects that didn't perform spiral
    targets= [labels[i]  for i,subject in enumerate(data) if len(subject[0])!=0]
    data=[subject for subject in data if len(subject[0])!=0]
    return data, targets
