# Training
import torch
import numpy as np
from scipy.signal import resample
from modules.utils import *
from modules.load_data import flip,rotate
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
clip=None, validation=False,window_size=None,task_i=None,augmentation=None,
paper_air_split=False,device="cuda",hierarchical=False,max_len=None):
    if batch_size!=1:
        raise NotImplementedError("batch_size should be 1, got {}".format(batch_size))
    losses=[]
    predictions=[]
    condition_targets=[]

    if augmentation is not None:
        if hierarchical:
            raise NotImplementedError("augmentation is not implemented for hierarchical models, see epoch function")
        #i is subject index and j is tranformation index
        super_index=[(index,j) for index in random_index for j in range(4)]# 4 because 3 transforms + original

        np.random.seed(1)
        np.random.shuffle(super_index)
        for index,j in super_index:
            condition_targets.append(targets[index])
            #augmentation
            subject=data[index].copy()
            #translation=np.random.uniform(0.5,1.5)
            #zoom_factor=np.random.uniform(0.8,1.2)
            crop=np.random.randint(len(subject)//10,len(subject)//5)#also worth for size of window warping
            window_warp=np.random.randint(0,len(subject)-crop)

            if j ==0:#keep original
                #warped=subject[window_warp:window_warp+crop]
                pass
            elif j==1:
                #apply translation/zoom to all measures but button_status
                #apply crop at the beginning
                #apply flip on x axis
                #UPSAMPLE worth both for rescaling and window warping
                rot=np.deg2rad(15)
                #subject*=zoom_factor#rotate(subject,rot)
                subject=upsample(subject)
                #subject[:,0]=-subject[:,0]
                #subject[:,measure2index["button_status"]]=data[index][:,measure2index["button_status"]]
            elif j==2:
                #apply flip on y axis
                #apply crop at the end
                #apply translation/zoom to all measures but spatial coordinate and button_status
                #DOWNSAMPLE *2 worth both for rescaling and window warping
                rot=np.deg2rad(-15)
                keep_measures=np.array([
                    measure2index['timestamp'],
                    measure2index['tilt'],
                    measure2index['elevation'],
                    measure2index['pressure'],
                    measure2index['speed'],
                    measure2index['acceleration']
                ])
                subject=downsample(subject,2)
                #subject[-crop:]=0
                #subject[keep_measures]*=zoom_factor#rotate(subject,rot)
            elif j==3:
                #apply flip both on x and y axis
                #apply crop both at the end and at the beginning
                #apply translation/zoom to spatial coordinate only
                #DOWNSAMPLE *4 worth both for rescaling and window warping
                subject=downsample(subject,4)
                rot=np.deg2rad(30)
                #subject[:,0]+=translation#*=zoom_factor
                keep_measures=np.array([
                    measure2index['y-coordinate'],
                    measure2index['x-coordinate']
                ])
                #subject[keep_measures]*=zoom_factor#translation#rotate(subject,rot)
            else:
                raise ValueError("expected j in range(4), got {}".format(j))
            #subject=np.concatenate((subject[:window_warp],warped,subject[window_warp+crop:]))
            if max_len is not None:
                subject=np.concatenate((subject,np.zeros(shape=(max_len-len(subject),subject.shape[1]))))
            #numpy to tensor
            target=torch.Tensor([targets[index]]).unsqueeze(0)
            if model.__class__.__name__!='Encoder' and model.__class__.__name__!= 'Model':
                if hierarchical:
                    subject=[torch.Tensor(seq.copy()).unsqueeze(0).transpose(1,2).to(device) for seq in subject]
                else:
                    subject=torch.Tensor(subject).unsqueeze(0).transpose(1,2)
            else:
                subject=torch.Tensor(subject).unsqueeze(1)#add batch dimension
            loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size,clip,validation,device=device,hierarchical=hierarchical)
            predictions.append(prediction)
            losses.append(loss)
    elif (task_i is not None and window_size is None and not paper_air_split) or hierarchical:#single task learning on the whole sequence
        for index in random_index:
            condition_targets.append(targets[index])
            if max_len is not None:
                subject=np.concatenate((data[index],np.zeros(shape=(max_len-len(data[index]),data[index].shape[1]))))
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
    #multitask learning (early fusion) OR single task learning on subsequences (either fixed window size or strokes)
    elif task_i is None or window_size is not None or (paper_air_split and not hierarchical):
        #if multitask learning len(data[i]) == 8 because 8 tasks
        super_index=[(i,j) for i in random_index for j in range(len(data[i]))]
        np.random.seed(1)
        np.random.shuffle(super_index)
        if window_size is not None or paper_air_split: #subsequences => we need to save predictions for late fusion (e.g. voting)
            predictions=dict(zip(random_index,[[] for _ in random_index]))
            condition_targets=[targets[i] for i in random_index]
        for i,j in super_index:#subject index, task index OR subsequence index
            if window_size is None and not paper_air_split:#we don't use the dictionary system so we have to keep track of the labels
                condition_targets.append(targets[i])
            #is it a on_paper or in-air stroke ?
            on_paper= data[i][j][0][measure2index["button_status"]]==on_paper_value
            #numpy to tensor
            #and add batch dimension
            if model.__class__.__name__!='Encoder' and model.__class__.__name__!= 'Model':
                subject=torch.Tensor(data[i][j]).unsqueeze(0).transpose(1,2)
            else:
                subject=torch.Tensor(data[i][j]).unsqueeze(1)

            target=torch.Tensor([targets[i]]).unsqueeze(0)
            #/!\ uncomment this to use the same model regardless of on paper or in air strokes
            #loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size, clip,validation,device=device,hierarchical=hierarchical)


            if paper_air_split and not on_paper:
                loss, prediction =step(subject,target, in_air,in_air_optimizer, loss_fn, batch_size,clip,validation,device=device,hierarchical=hierarchical)
            else:
                loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size, clip,validation,device=device,hierarchical=hierarchical)

            if window_size is not None or paper_air_split: #subsequences => we need to save predictions for late fusion (e.g. voting)
                predictions[i].append(prediction)
            else:#no late fusion => we just care about the label
                predictions.append(prediction)
            losses.append(loss)
    else:
        raise NotImplementedError("check readme or code.")

    if window_size is not None or (paper_air_split and not hierarchical): #subsequences => we need fuse the predictions of each sub seq (e.g. voting)
        #average over each model's prediction : choose between this and majority voting
        predictions=[np.mean(sub) for sub in list(predictions.values())]

        #majority voting : choose between this and average fusion
        #predictions=[np.mean(list(map(round,sub))) for sub in list(predictions.values())]

    return condition_targets,predictions,np.mean(losses)#[np.mean(losses),accuracy,sensitivity,specificity,ppv,npv],false
