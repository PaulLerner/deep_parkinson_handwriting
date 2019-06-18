# Training
import torch
import numpy as np
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
    if not hierarchical:#if hierarchical the device option is in the forward function
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
        if clip is not None:
            #clip encoder gradients to previent exploding
            if decoder:#model is encoder
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            elif hierarchical:
                torch.nn.utils.clip_grad_norm_(model.layer1.parameters(), clip)
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
clip=None, validation=False,window_size=None,task_i=None,augmentation=False,
paper_air_split=False,device="cuda",hierarchical=False):
    losses=[]
    predictions=[]
    condition_targets=[]

    if augmentation:
        #i is subject index and j is tranformation index
        super_index=[(index,j) for index in random_index for j in range(4)]# 4 because 3 transforms + original

        np.random.seed(1)
        np.random.shuffle(super_index)
        for index,j in super_index:
            condition_targets.append(targets[index])
            #augmentation
            subject=data[index].copy()
            """if hierarchical:
                for k,sub in enumerate(subject):
                    subject[k][:,4:]+=np.random.randn(sub.shape[0],3)*1e-2
            else:
                subject[:,4:]+=np.random.randn(subject.shape[0],3)*1e-2"""
            translation=np.random.rand()-0.5
            if j ==0:#keep original
                pass
            elif j==1:
                subject[:,0]+=translation#flip(data[index].copy(),axis_i=0)
            elif j==2:
                subject[:,1]+=translation#rotate(data[index].copy(),np.deg2rad(-15))
            elif j==3:
                subject[:,0]+=translation#*=zoom_factor
                subject[:,1]+=translation
            else:
                raise ValueError("expected j in range(4), got {}".format(j))
            #numpy to tensor
            target=torch.Tensor([targets[index]]).unsqueeze(0)
            if model.__class__.__name__!='Encoder' or model.__class__.__name__!= 'Model':
                if hierarchical:
                    subject=[torch.Tensor(seq.copy()).unsqueeze(0).transpose(1,2).to(device) for seq in subject]
                else:
                    subject=torch.Tensor(subject).unsqueeze(0).transpose(1,2)
            else:
                subject=torch.Tensor(subject).unsqueeze(1)#add batch dimension


            loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size,clip,validation,device=device,hierarchical=hierarchical)
            predictions.append(round(prediction))
            losses.append(loss)
    elif task_i is not None and window_size is None and not paper_air_split:#single task learning on the whole sequence
        if batch_size==1:
            for index in random_index:
                condition_targets.append(targets[index])
                #numpy to tensor
                if hierarchical:
                    if model.__class__.__name__!='Encoder' or model.__class__.__name__!= 'Model':
                        subject=[torch.Tensor(seq.copy()).unsqueeze(0).transpose(1,2).to(device) for seq in data[index]]
                    else:
                        subject=[torch.Tensor(seq.copy()).unsqueeze(1).to(device) for seq in data[index]]
                elif model.__class__.__name__!='Encoder' or model.__class__.__name__!= 'Model':
                    subject=torch.Tensor(data[index].copy()).unsqueeze(0).transpose(1,2)
                else:
                    subject=torch.Tensor(data[index].copy()).unsqueeze(1)#add batch dimension
                target=torch.Tensor([targets[index]]).unsqueeze(0)

                loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size,clip,validation,device=device,hierarchical=hierarchical)
                predictions.append(round(prediction))
                losses.append(loss)
        else:
            condition_targets=targets[random_index]
            tensor_data=torch.Tensor(data[random_index]).transpose(1,2)
            tensor_targets=torch.Tensor(targets[random_index]).unsqueeze(1)
            losses, predictions =step(tensor_data,tensor_targets, model, optimizer, loss_fn, batch_size,clip,validation,device=device,hierarchical=hierarchical)
            predictions=list(map(round,predictions))

    #multitask learning (early fusion) OR single task learning on subsequences (either fixed window size or strokes)
    elif task_i is None or window_size is not None or paper_air_split:
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
            if model.__class__.__name__!='Encoder' or model.__class__.__name__!= 'Model':
                subject=torch.Tensor(data[i][j]).unsqueeze(0).transpose(1,2)
            else:
                subject=torch.Tensor(data[i][j]).unsqueeze(1)

            target=torch.Tensor([targets[i]]).unsqueeze(0)
            loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size, clip,validation,device=device,hierarchical=hierarchical)

            """#/!\ uncomment this to use different models when paper_air_split
            if paper_air_split and not on_paper:
                loss, prediction =step(subject,target, in_air,in_air_optimizer, loss_fn, batch_size,clip,validation, decoder, decoder_optimizer,device=device,hierarchical=hierarchical)
            else:
                loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size, clip,validation,decoder, decoder_optimizer,device=device,hierarchical=hierarchical)
            """
            if window_size is not None or paper_air_split: #subsequences => we need to save predictions for late fusion (e.g. voting)
                predictions[i].append(prediction)
            else:#no late fusion => we just care about the label
                predictions.append(round(prediction))
            losses.append(loss)
    else:
        raise NotImplementedError("check readme or code.")

    if window_size is not None or paper_air_split: #subsequences => we need fuse the predictions of each sub seq (e.g. voting)
        #average over each model's prediction : choose between this and majority voting
        predictions=[round(np.mean(sub)) for sub in list(predictions.values())]

        #majority voting : choose between this and average fusion
        #predictions=[round(np.mean(list(map(round,sub)))) for sub in list(predictions.values())]

    return condition_targets,predictions,np.mean(losses)#[np.mean(losses),accuracy,sensitivity,specificity,ppv,npv],false
