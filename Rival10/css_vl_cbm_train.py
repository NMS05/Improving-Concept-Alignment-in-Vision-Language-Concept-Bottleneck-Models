import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1111)
from torch.utils.data import DataLoader

from data.my_rival_dataset import CSS_Rival_Dataset
from model.cbm_models import css_cbm, trinity_loss

import time
import numpy as np



def estimate_top_concepts_accuracy(concept_predictions, concept_labels):

    bs = concept_predictions.shape[0]
    mini_batch_correct_concepts = 0
    mini_batch_total_concepts = 0

    for i in range(bs):

        k_val = int(torch.sum(concept_labels[i]).item())  # active labels for that class
        _, top_gt_indices = torch.topk(concept_labels[i], k=k_val, dim=-1)
        _, top_pred_indices = torch.topk(concept_predictions[i], k=k_val, dim=-1)

        for k in top_pred_indices:
            mini_batch_total_concepts+=1
            if k in top_gt_indices: mini_batch_correct_concepts+=1

    return mini_batch_correct_concepts, mini_batch_total_concepts


def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    
    contrastive_loss = []
    classifier_loss = []
    concept_loss = []

    sum_correct_pred = 0
    total_samples = 0
    concept_acc = 0
    concept_count = 0
    
    model.train()

    ###Iterating over data loader
    for i, (images, class_labels, concept_labels, use_concept_labels) in enumerate(train_data_loader):
        
        #Loading data and labels to device
        images = images.squeeze().to(device)
        class_labels = class_labels.squeeze().to(device)
        concept_labels = concept_labels.squeeze().to(device)
        use_concept_labels = use_concept_labels.squeeze().to(device)

        # stack the labels along batch dimension (no longer need to be in pairs)
        class_labels = torch.reshape(class_labels,(class_labels.shape[0]*2,1)).squeeze()
        concept_labels = torch.reshape(concept_labels,(concept_labels.shape[0]*2,18))
        use_concept_labels = torch.reshape(use_concept_labels,(use_concept_labels.shape[0]*2,1)).squeeze()

        #Reseting Gradients
        optimizer.zero_grad()

        #Forward
        concept_predictions, class_predictions = model(images)

        #Calculating Loss
        loss1, loss2, loss3 = loss_fn(concept_predictions, class_predictions, class_labels, concept_labels, use_concept_labels)
        _loss = loss1 + loss2 + loss3

        contrastive_loss.append(loss1.item())
        classifier_loss.append(loss2.item())
        concept_loss.append(loss3.item())

        #Backward
        _loss.backward()
        optimizer.step()

        if i%200 == 0:
            print("Contrastive Loss = ",np.mean(contrastive_loss), ", Classifier Loss = ",np.mean(classifier_loss),", Concept Loss = ", np.mean(concept_loss))
        
        # calculate acc per minibatch
        sum_correct_pred += (torch.argmax(class_predictions, dim=-1) == class_labels).sum().item()
        total_samples += len(class_labels)
        mbcc, mbtc = estimate_top_concepts_accuracy(concept_predictions, concept_labels)
        concept_acc += mbcc
        concept_count += mbtc


    acc = round(sum_correct_pred/total_samples,4)*100
    total_concept_acc = round(concept_acc/concept_count,4)*100
    return acc, total_concept_acc


def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    contrastive_loss = []
    classifier_loss = []
    concept_loss = []

    sum_correct_pred = 0
    total_samples = 0
    concept_acc = 0
    concept_count = 0

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, (images, class_labels, concept_labels, use_concept_labels) in enumerate(val_data_loader):
            
            #Loading data and labels to device
            images = images.squeeze().to(device)
            class_labels = class_labels.squeeze().to(device)
            concept_labels = concept_labels.squeeze().to(device)
            use_concept_labels = use_concept_labels.squeeze().to(device)

            # stack the labels along batch dimension (no longer need to be in pairs)
            class_labels = torch.reshape(class_labels,(class_labels.shape[0]*2,1)).squeeze()
            concept_labels = torch.reshape(concept_labels,(concept_labels.shape[0]*2,18))
            use_concept_labels = torch.reshape(use_concept_labels,(use_concept_labels.shape[0]*2,1)).squeeze()

            #Forward
            concept_predictions, class_predictions = model(images)

            #Calculating Loss
            loss1, loss2, loss3 = loss_fn(concept_predictions, class_predictions, class_labels, concept_labels, use_concept_labels)

            contrastive_loss.append(loss1.item())
            classifier_loss.append(loss2.item())
            concept_loss.append(loss3.item())
            
            # calculate acc per minibatch
            sum_correct_pred += (torch.argmax(class_predictions, dim=-1) == class_labels).sum().item()
            total_samples += len(class_labels)
            mbcc, mbtc = estimate_top_concepts_accuracy(concept_predictions, concept_labels)
            concept_acc += mbcc
            concept_count += mbtc

    acc = round(sum_correct_pred/total_samples,4)*100
    total_concept_acc = round(concept_acc/concept_count,4)*100
    return acc, total_concept_acc


def train_clip(batch_size, epochs, device):

    # DataLoader
    train_dataset = CSS_Rival_Dataset(split="train",true_batch_size=batch_size, percentage_of_concept_labels_for_training=0.01)
    train_loader = DataLoader(train_dataset, batch_size=1,num_workers=16)
    test_dataset = CSS_Rival_Dataset(split="test",true_batch_size=batch_size, percentage_of_concept_labels_for_training=0.0)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=16)
      

    # Model and Loss
    model = css_cbm()
    model.to(device)
    print("\n\n\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Train
    loss_fn = trinity_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    print("\n\t Started Training\n")


    for epoch in range(epochs):

        begin = time.time()

        ###Training
        acc, concept_acc = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        ###Validation
        val_acc, val_concept_acc = val_one_epoch(test_loader, model, loss_fn, device)

        print('\n\t Epoch....', epoch + 1)
        print("\t Train Class Accuracy = {} and Train Concept Accuracy = {}.".format(round(acc,2),round(concept_acc,2)))
        print("\t Val Class Accuracy = {} and Val Concept Accuracy = {}.".format(round(val_acc,2),round(val_concept_acc,2)))
        print('\t Time per epoch (in mins) = ', round((time.time()-begin)/60,2),'\n\n')


if __name__=="__main__":
    train_clip(batch_size=8, epochs=5, device = torch.device("cuda"))