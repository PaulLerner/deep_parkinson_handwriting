import numpy as np
from time import time
import matplotlib.pyplot as plt

measure2index={"y-coordinate":0,"x-coordinate":1,"timestamp":2, "button_status":3,"tilt":4, "elevation":5,"pressure":6,
"speed":7,"acceleration":8}
index2measure=list(measure2index.keys())

task2index={"spiral":0,"l":1,"le":2 ,"les":3,"lektorka" :4,"porovnat":5,"nepopadnout":6, "tram":7}
index2task=list(task2index.keys())

max_lengths=[16071, 4226, 6615, 6827, 7993, 5783, 4423, 7676]

plot2index={"loss":0,"accuracy":1}
index2plot= list(plot2index.keys())
on_paper_value=1.0#on_paper_stroke iff button_status==1.0
one_hot=np.identity(8)


def get_out_size(in_size,padding,dilation,kernel_size,stride):
    """computes output size after a conv or a pool layer"""
    return (in_size+2*padding-dilation*(kernel_size-1)-1)//stride +1
def min_max_scale(data,min_=0,max_=1):
    return (max_-min_)*(data-np.min(data)/(np.max(data)-np.min(data)))+min_
def count_params(model):
    """returns (total n° of parameters, n° of trainable parameters)"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
def plot_task(task):
    plt.plot(task[:,1],task[:,0])
    plt.xlabel(index2measure[1])
    plt.ylabel(index2measure[0])
def plot_measures(task):
    plt.figure(figsize=(16,12))
    for i,measure in enumerate(index2measure[:-2]):
        plt.subplot(3,3,i+1)
        plt.plot(task[:,i])
        plt.xlabel("timesteps")
        plt.ylabel(measure)
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
def ReshapeAndVote(model_train_predictions):
    """used to fuse the predictions of n_models models after n_CV CV"""
    n_CV=len(model_train_predictions[0])
    n_models=len(model_train_predictions)
    reshaped_train_predictions=[[model_train_predictions[i][j] for i in range(n_models)] for j in range(n_CV)]

    voted_train_predictions=[np.around(np.mean(reshaped_train_predictions[i],axis=0)) for i in range(n_CV)]
    return voted_train_predictions

def confusion_matrix(y_true,y_pred):
    if len(y_true)!=len(y_pred):
        raise ValueError("y_true and y_pred should have the same shape, got {} and {}, respectively".format(len(y_true),len(y_pred)))
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
