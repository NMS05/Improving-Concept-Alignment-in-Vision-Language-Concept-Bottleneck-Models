"""
This snippet allows you to calculate the classwise error matrix during inference stage
From  the error matrix, you can find the confounding classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1111)
from torch.utils.data import DataLoader

import open_clip
import numpy as np

from data.cub_data import Processed_CUB_Dataset


class trained_cbm(nn.Module):
    def __init__(self,):
        super(trained_cbm, self).__init__()

        # clip model
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
        self.clip_model.visual.output_tokens = True
        # throw away text tower to save memory
        self.clip_model.transformer = nn.Identity()
        self.clip_model.token_embedding = nn.Identity()
        self.clip_model.ln_final = nn.Identity()
        # Freeze params
        for p in self.clip_model.parameters(): p.requires_grad=False
        
        # text concept features; shape = (312,512)
        self.concept_features = np.load("model/my_cub_concept_features.npy")

        self.concept_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768,312)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(312),
            nn.Linear(312,200),
        )
    
    def _get_concept_tensor(self, device):
        return torch.tensor(self.concept_features, requires_grad=False, device=device)
    
    def forward(self, images):
        visual_projection, visual_patches = self.clip_model.encode_image(images)
        concept_activations = F.normalize(visual_projection, dim=-1) @ F.normalize(self._get_concept_tensor(device=visual_projection.device),dim=-1).T
        concept_projections = self.concept_projection(torch.mean(visual_patches,dim=1))
        concepts = concept_activations + concept_projections
        return concepts, self.classifier(concepts)


def val_one_epoch(val_data_loader, model, device):
    
    ### Local Parameters
    sum_correct_pred = 0
    total_samples = 0

    # initialize error matrix
    error_matrix = torch.zeros(200,200)

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, (images, class_labels) in enumerate(val_data_loader):
            
            #Loading data and labels to device
            images = images.to(device)
            class_labels = class_labels.to(device)

            #Forward
            concept_predictions, class_predictions = model(images)
            
            # calculate acc per minibatch
            sum_correct_pred += (torch.argmax(class_predictions, dim=-1) == class_labels).sum().item()
            total_samples += len(class_labels)

            ##### calculate the classwise error matrix
            predicted_classes = torch.argmax(class_predictions, dim=-1)
            for x in range(predicted_classes.shape[0]): # for every sample in a mini-batch
                if predicted_classes[x] != class_labels[x]: #for incorrect predictions
                    idx1 = class_labels[x].item() # class under investigation
                    idx2 = predicted_classes[x].item() # it is often mis-identified as
                    error_matrix[idx1,idx2] += 1.
    
    overall_classification_accuracy = round(sum_correct_pred/total_samples,4)*100
    return overall_classification_accuracy, error_matrix


def validate_cbm(batch_size, device):

    # DataLoader
    test_dataset = Processed_CUB_Dataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    # Model
    model = trained_cbm()
    model.load_state_dict(torch.load("trained_css_vl_cbm.pth"))
    for p in model.parameters(): p.requires_grad = False
    model.to(device)
    print("\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    val_acc, matrix = val_one_epoch(test_loader, model, device)
    print("\n\t Test Accuracy = ", val_acc,"\n")

    np.save('error_matrix.npy',matrix.numpy())

if __name__=="__main__":
    validate_cbm(batch_size=128, device = torch.device("cuda:0"))