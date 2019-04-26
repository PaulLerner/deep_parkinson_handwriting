# Training
import torch
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)
## step

def step(input, target, model, optimizer, loss_fn, batch_size,clip=None,validation = False):
    if not validation:
        # Zero gradients
        optimizer.zero_grad()

    # Set device options
    input=input.to(device)
    target=target.to(device)

    #forward pass
    output=model(input)

    # Compute loss
    loss = loss_fn(output, target)
    if not validation:
        # Perform backpropagation
        loss.backward()
        if clip is not None:
            #clip gradients to previent exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Adjust model weights
        optimizer.step()
        """# carry over regularization # useless if we reset hidden state for each subject
        if np.random.rand() <carry_over:
            model.reset_hidden() """
    #reset hidden state after each step (i.e. after each subject OR each task OR each subsequence)
    model.reset_hidden()

    """#continuous learning : choose between this and `model.reset_hidden() `
    #early results suggest that continuous learning make the loss unstable

    #we pass the hidden state from subject to subject !
    #but we detach it because we can't backprop through the whole dataset
    model.hidden_state=model.hidden_state.detach()
    if is_lstm:
        model.cell_state=model.cell_state.detach()"""

    return loss.item(), output.item()

## epoch

def epoch(data,targets, model, optimizer, loss_fn, batch_size, random_index,clip=None,
          validation=False,window_size=None,task_i=None):
    losses=[]
    predictions=[]
    condition_targets=[]

    if task_i is not None and window_size is None:#single task learning on the whole sequence
        for index in random_index:
            condition_targets.append(targets[index])
            #numpy to tensor
            subject=torch.Tensor(data[index]).unsqueeze(1)#add batch dimension
            target=torch.Tensor([targets[index]])

            loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size,clip,validation)
            predictions.append(round(prediction))
            losses.append(loss)
    else:#multitask learning (early fusion) OR single task learning on subsequences
        #if multitask learning len(data[i]) == 8 because 8 tasks
        super_index=[(i,j) for i in random_index for j in range(len(data[i]))]
        np.random.shuffle(super_index)
        if window_size is not None: #subsequences => we need to save predictions for late fusion (e.g. voting)
            predictions=dict(zip(random_index,[[] for _ in random_index]))
            condition_targets=[targets[i] for i in random_index]
        for i,j in super_index:#subject index, task index OR subsequence index
            if window_size is None:#we don't use the dictionary system so we have to keep track of the labels
                condition_targets.append(targets[i])
            #numpy to tensor
            #and add batch dimension
            subject=torch.Tensor(data[i][j]).unsqueeze(1)
            target=torch.Tensor([targets[i]])
            loss, prediction =step(subject,target, model, optimizer, loss_fn, batch_size,clip,validation)
            if window_size is not None: #subsequences => we need to save predictions for late fusion (e.g. voting)
                predictions[i].append(prediction)
            else:#no late fusion => we just care about the label
                predictions.append(round(prediction))
            losses.append(loss)

    if window_size is not None: #subsequences => we need fuse the predictions of each sub seq (e.g. voting)
        #average over each model's prediction : choose between this and majority voting
        predictions=[round(np.mean(sub)) for sub in list(predictions.values())]

        #majority voting : choose between this and average fusion
        #predictions=[round(np.mean(list(map(round,sub)))) for sub in list(predictions.values())]

    #compute metrics
    tn, fp, fn, tp, false_i = confusion_matrix(y_true=condition_targets,y_pred=predictions)
    if task_i is not None:
        false=[random_index[i] for i in false_i]
    else:
        false=[random_index[i%len(random_index)] for i in false_i]
    accuracy= (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn) if (tp+fn) != 0 else 0.0 #without condition positives the sensitivity should be 0
    specificity = tn/(tn+fp) if (tn+fp)!= 0 else 0.0 #idem
    ppv = tp/(tp+fp) if tp+fp != 0 else 0.0 #without predicted positives the ppv should be 0
    npv = tn/(tn+fn) if tn+fn !=0 else 0.0 #idem

    return [np.mean(losses),accuracy,sensitivity,specificity,ppv,npv],false
