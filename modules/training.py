# Training
import torch
import numpy as np
from scipy.signal import resample
from .utils import *
from .load_data import flip,rotate
## step

def step(input, target, model, optimizer, loss_fn, batch_size, clip=None,validation = False, decoder=None,
         decoder_optimizer = None,device="cuda",hierarchical=False):
    if not validation:
        # Zero gradients
        optimizer.zero_grad()
        if decoder_optimizer:
            decoder_optimizer.zero_grad()

    # Set device options
    target=target.to(device)
    if not hierarchical:#if hierarchical the device option is done in epoch
        input=input.to(device)

    #forward pass
    output=model(input)
    if decoder:
        output=decoder(output).squeeze(0)

    # Compute loss
    loss = loss_fn(output, target)
    if not validation:
        # Perform backpropagation
        loss.backward()
        name=model.__class__.__name__
        if clip is not None and (name=='Encoder' or name== 'Model' or name=="HierarchicalStrokeCRNN"):
            #clip encoder gradients to previent exploding
            if decoder:#model is encoder
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            elif hierarchical:
                #torch.nn.utils.clip_grad_norm_(model.layer1.parameters(), clip)
                #torch.nn.utils.clip_grad_norm_(model.layer2.parameters(), clip)
                if name=="HierarchicalStrokeCRNN":
                    torch.nn.utils.clip_grad_norm_(model.rnn.parameters(), clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.layer1.parameters(), clip)
                    torch.nn.utils.clip_grad_norm_(model.layer1_air.parameters(), clip)
                    torch.nn.utils.clip_grad_norm_(model.layer2.parameters(), clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), clip)
        # Adjust model weights
        optimizer.step()
        if decoder_optimizer:
            decoder_optimizer.step()
    if model.__class__.__name__=='Encoder' or model.__class__.__name__== 'Model':
        #reset hidden state after each step (i.e. after each subject OR each task OR each subsequence)
        model.reset_hidden(device)
    if batch_size==1:
        return loss.item(), output.item()
    else:
        return loss.item(), output.squeeze().cpu().detach().numpy()
## epoch

def epoch(data,targets, model, optimizer, loss_fn, batch_size, random_index,
in_air =None, in_air_optimizer=None,decoder=None,decoder_optimizer=None,
clip=None, validation=False,window_size=None,task_i=None,
paper_air_split=False,device="cuda",hierarchical=False,max_len=None):
    if batch_size!=1:
        raise NotImplementedError("batch_size should be 1, got {}".format(batch_size))
    losses=[]
    predictions=[]
    condition_targets=[]

    if not paper_air_split or hierarchical:#single task learning on the whole sequence OR hierarchical model
        for index in random_index:
            condition_targets.append(targets[index])
            if max_len is not None:
                subject=np.concatenate((data[index],np.zeros(shape=(max_len-len(data[index]),data[index].shape[1]))))
            else:
                subject=data[index]
            #numpy to tensor
            if hierarchical:
                if model.__class__.__name__!='Encoder' and model.__class__.__name__!= 'Model' and model.__class__.__name__!= 'HierarchicalRNN':
                    subject=[torch.Tensor(seq.copy()).unsqueeze(0).transpose(1,2).to(device) for seq in subject]
                else:
                    subject=[torch.Tensor(seq.copy()).unsqueeze(1).to(device) for seq in subject]
            elif model.__class__.__name__!='Encoder' and model.__class__.__name__!= 'Model':
                subject=torch.Tensor(subject.copy()).unsqueeze(0).transpose(1,2)
            else:
                subject=torch.Tensor(subject.copy()).unsqueeze(1)#add batch dimension
            target=torch.Tensor([targets[index]]).unsqueeze(0)

            loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size,clip,validation,device=device,hierarchical=hierarchical)
            predictions.append(prediction)
            losses.append(loss)
            #foo2
    #single task learning on subsequences (e.g. strokes)
    elif paper_air_split and not hierarchical:
        #if multitask learning len(data[i]) == 8 because 8 tasks
        super_index=[(i,j) for i in random_index for j in range(len(data[i]))]
        np.random.seed(1)
        np.random.shuffle(super_index)
        predictions=dict(zip(random_index,[[] for _ in random_index]))
        condition_targets=[targets[i] for i in random_index]
        for i,j in super_index:#subject index, subsequence index
            if in_air is not None:#is it a on_paper or in-air stroke ?
                on_paper= data[i][j][0][measure2index["button_status"]]==on_paper_value
            else:
                on_paper=False #we don't care, it just has to be a bool
            if max_len is not None:
                subject=np.concatenate((data[i][j],np.zeros(shape=(max_len-len(data[i][j]),data[i][j].shape[1]))))
            else:
                subject=data[i][j]
            #numpy to tensor
            #and add batch dimension
            if model.__class__.__name__!='Encoder' and model.__class__.__name__!= 'Model':
                subject=torch.Tensor(subject).unsqueeze(0).transpose(1,2)
            else:
                subject=torch.Tensor(subject).unsqueeze(1)

            target=torch.Tensor([targets[i]]).unsqueeze(0)

            if not on_paper and in_air is not None:
                loss, prediction =step(subject,target, in_air,in_air_optimizer, loss_fn, batch_size,clip,validation,device=device,hierarchical=hierarchical)
            else:
                loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size, clip,validation,device=device,hierarchical=hierarchical)

            predictions[i].append(prediction)
            losses.append(loss)
    else:
        raise NotImplementedError("check readme or code.")

    if paper_air_split and not hierarchical: #subsequences => we need fuse the predictions of each sub seq (e.g. voting)
        #average over each model's prediction : choose between this and majority voting
        predictions=[np.mean(sub) for sub in list(predictions.values())]

        #majority voting : choose between this and average fusion
        #predictions=[np.mean(list(map(round,sub))) for sub in list(predictions.values())]

    return condition_targets,predictions,np.mean(losses)#[np.mean(losses),accuracy,sensitivity,specificity,ppv,npv],false
