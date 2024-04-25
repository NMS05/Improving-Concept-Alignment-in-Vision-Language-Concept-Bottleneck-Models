import torch
import torchvision
import torch.nn as nn
import open_clip
import numpy as np


"""
A simple VL-CBM model without concept projection layer
"""
class clip_cbm(nn.Module):
    def __init__(self):
        super(clip_cbm, self).__init__()

        # clip model
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
        self.clip_model.transformer = nn.Identity() # throw away text tower to save memory
        self.clip_model.token_embedding = nn.Identity()
        self.clip_model.ln_final = nn.Identity()
        for p in self.clip_model.parameters(): p.requires_grad=False

        # concept features as tensors. shape = (312,512). My concepts
        self.concept_features = np.load("model/my_cub_concept_features.npy")

        # classifer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(312),
            nn.Linear(312,200)
        )
    
    def _get_concept_tensor(self, device):
        return torch.tensor(self.concept_features, requires_grad=False, device=device)
    
    def forward(self, images): 
        visual_features = self.clip_model.encode_image(images, normalize=True) #(bs,512)
        dev = visual_features.device
        concept_activations = visual_features @ self._get_concept_tensor(device=dev).T #(bs,312)
        return concept_activations, self.classifier(concept_activations)