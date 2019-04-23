import numpy as np

measure2index={"y-coordinate":0,"x-coordinate":1,"timestamp":2, "button_status":3,"tilt":4, "elevation":5,"pressure":6}
index2measure=list(measure2index.keys())

task2index={"spiral":0,"l":1,"le":2 ,"les":3,"lektorka" :4,"porovnat":5,"nepopadnout":6, "tram":7}
index2task=list(task2index.keys())

one_hot=np.identity(8)

def timeSince(since):
    now = time()
    s = now - since
    m = np.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def return_results(train_metrics,valid_metrics,early_stopping,flat_falses):
    train_metrics,valid_metrics=np.asarray(train_metrics),np.asarray(valid_metrics)
    model_name="LSTM" if is_lstm else "GRU"
    task_name=index2task[task_i] if task_i is not None else str(task_i)
    results="{} ; {} ; {} ; {} ; {} ; {} ; {} ; {} ; {} ; {} ; {} ; {:.2f} (+ {:.2f}) ; {:.2f} (+ {:.2f}) ".format(
    task_name,model_name,compute_movement,downsampling_factor,learning_rate,
        hidden_size,num_layers,bidirectional,carry_over,dropout,clip,
    np.mean(early_stopping),np.std(early_stopping),np.mean(train_metrics[:,1]),np.std(train_metrics[:,1]))
    
    valid_metrics=valid_metrics.T
    for metric in valid_metrics[1:]:#don't care about the loss
        mean,std=np.mean(metric),np.std(metric)
        results+="; {:.2f} (+ {:.2f}) ".format(mean,std)
    results+=" ; "
    results+=" ; ".join(map(str, flat_falses))
    return results

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

def plot_loss():
    print(results[0:80].replace(";","|"))
    plt.figure()
    plt.title("average loss over 10 folds over the {} first epochs".format(shortest_fold))
    plt.plot(avg_train[:,0],label="training")
    plt.plot(avg_valid[:,0],label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
