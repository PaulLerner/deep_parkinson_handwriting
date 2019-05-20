import numpy as np
from time import time
import matplotlib.pyplot as plt
measure2index={"y-coordinate":0,"x-coordinate":1,"timestamp":2, "button_status":3,"tilt":4, "elevation":5,"pressure":6,
"speed":7,"acceleration":8}
index2measure=list(measure2index.keys())

task2index={"spiral":0,"l":1,"le":2 ,"les":3,"lektorka" :4,"porovnat":5,"nepopadnout":6, "tram":7}
index2task=list(task2index.keys())

plot2index={"loss":0,"accuracy":1}
index2plot= list(plot2index.keys())
on_paper_value=1.0#on_paper_stroke iff button_status==1.0
one_hot=np.identity(8)

def plot_task(task):
    plt.plot(task[:,1],task[:,0])
    plt.xlabel(index2measure[1])
    plt.ylabel(index2measure[0])
def return_metrics(tp,tn,fp,fn):
    accuracy= (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn) if (tp+fn) != 0 else 0.0 #without condition positives the sensitivity should be 0
    specificity = tn/(tn+fp) if (tn+fp)!= 0 else 0.0 #idem
    ppv = tp/(tp+fp) if tp+fp != 0 else 0.0 #without predicted positives the ppv should be 0
    npv = tn/(tn+fn) if tn+fn !=0 else 0.0 #idem
    return accuracy,sensitivity,specificity,ppv,npv
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def flat_list(list):
    return [item for sublist in list for item in sublist]

def timeSince(since):
    now = time()
    s = now - since
    m = np.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def confusion_matrix(y_true,y_pred):
    tn, fp, fn, tp=0,0,0,0
    false_i=[]
    for i, (target, pred) in enumerate(list(zip(y_true,y_pred))):
        if target==0:#condition negative
            if pred==0:
                tn+=1
            else:
                fp+=1
                false_i.append(i)
        else:#condition positive
            if pred==0:
                fn+=1
                false_i.append(i)
            else:
                tp+=1
    return tn, fp, fn, tp, false_i
