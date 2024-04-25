import torch
import torch.nn as nn
torch.manual_seed(1111)
from torch.utils.data import DataLoader

from data.cub_data import Processed_CUB_Dataset  
from model.cbms import clip_cbm

import time
import numpy as np


###############################################################################################################
#                                           Estimating concept accuracy
###############################################################################################################

# cacs = class-averaged concept scores
cacs = torch.tensor(np.load("data/class_averaged_concept_scores.npy"))

def get_cacs(device, label):
    _cacs = cacs.to(device)
    return _cacs[label] #(bs,312)

def estimate_top_concepts_accuracy(concept_predictions, label):
    bs = concept_predictions.shape[0]
    ground_truth_concepts = get_cacs(concept_predictions.device,label) #(bs,312)

    mini_batch_correct_concepts = 0
    mini_batch_total_concepts = 0

    for i in range(bs):      
        _, top_gt_indices = torch.topk(ground_truth_concepts[i], k=32, dim=-1)
        _, top_pred_indices = torch.topk(concept_predictions[i], k=32, dim=-1)
        for k in top_pred_indices:
            mini_batch_total_concepts+=1
            if k in top_gt_indices: mini_batch_correct_concepts+=1
    
    return mini_batch_correct_concepts, mini_batch_total_concepts



###############################################################################################################
#                              Snippet to train CLIP VL-CBM (no concept supervision)
###############################################################################################################

def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    
    classifier_loss = []
    sum_correct_pred = 0
    total_samples = 0
    concept_acc = 0
    concept_count = 0
    
    model.train()

    ###Iterating over data loader
    for i, (images, labels) in enumerate(train_data_loader):
        
        #Loading data and labels to device
        images = images.to(device)
        labels = labels.to(device)

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        concept_preds, preds = model(images)
        #Calculating Loss
        _loss = loss_fn(preds, labels)
        classifier_loss.append(_loss.item())      
        #Backward
        _loss.backward()
        optimizer.step()

        if i%50 == 0: print("train_loss = ", np.mean(classifier_loss))
        
        # calculate acc per minibatch
        sum_correct_pred += (torch.argmax(preds, dim=-1) == labels).sum().item()
        total_samples += len(labels)
        mbcc, mbtc = estimate_top_concepts_accuracy(concept_preds, labels)
        concept_acc += mbcc
        concept_count += mbtc

    ###Acc and Loss
    acc = round(sum_correct_pred/total_samples,4)*100
    total_concept_acc = round(concept_acc/concept_count,4)*100
    epoch_loss = np.mean(classifier_loss)
    return acc, total_concept_acc, epoch_loss
    

def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    classifier_loss = []
    sum_correct_pred = 0
    total_samples = 0
    concept_acc = 0
    concept_count = 0

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, (images, labels) in enumerate(val_data_loader):
            
            #Loading data and labels to device
            images = images.to(device)
            labels = labels.to(device)

            #Forward
            concept_preds, preds = model(images)
            #Calculating Loss
            _loss = loss_fn(preds, labels)
            classifier_loss.append(_loss.item())
            
            # calculate acc per minibatch
            sum_correct_pred += (torch.argmax(preds, dim=-1) == labels).sum().item()
            total_samples += len(labels)
            mbcc, mbtc = estimate_top_concepts_accuracy(concept_preds, labels)
            concept_acc += mbcc
            concept_count += mbtc

    ###Acc and Loss
    acc = round(sum_correct_pred/total_samples,4)*100
    total_concept_acc = round(concept_acc/concept_count,4)*100
    epoch_loss = np.mean(classifier_loss)
    return acc, total_concept_acc, epoch_loss


def train_clip(batch_size, epochs):
    """
    DataLoader
    """
    train_dataset = Processed_CUB_Dataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataset = Processed_CUB_Dataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
      
    """
    Model and Loss
    """
    model = clip_cbm()
    device = torch.device("cuda")
    model.to(device)
    print("\n\n\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    """
    Train
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    print("\n\t Started Training\n")

    for epoch in range(epochs):

        begin = time.time()

        ###Training
        train_class_acc, train_concept_acc, train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        ###Validation
        val_class_acc, val_concept_acc, val_loss = val_one_epoch(test_loader, model, loss_fn, device)

        print('\n\t Epoch....', epoch + 1)
        print("\t Train Class Accuracy = {} and Train Concept Accuracy = {}.".format(round(train_class_acc,2),round(train_concept_acc,2)))
        print("\t Val Class Accuracy = {} and Val Concept Accuracy = {}.".format(round(val_class_acc,2),round(val_concept_acc,2)))
        print('\t Time per epoch (in mins) = ', round((time.time()-begin)/60,2),'\n\n')


if __name__=="__main__":
    train_clip(batch_size=64, epochs=20)