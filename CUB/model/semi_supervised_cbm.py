import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
import numpy as np


"""
###############################################################################################################
                                                        MODEL
###############################################################################################################
"""


# Contrastive Semi-Supervised (CSS) VL-CBM
class css_cbm(nn.Module):
    def __init__(self,):
        super(css_cbm, self).__init__()

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
    
    def forward(self, image_pairs):

        bs,imgs,channels,h,w = image_pairs.shape
        images = torch.reshape(image_pairs,(bs*imgs,channels,h,w))

        visual_projection, visual_patches = self.clip_model.encode_image(images) #(bs*2,512),(bs*2,196,768)
        concept_activations = F.normalize(visual_projection, dim=-1) @ F.normalize(self._get_concept_tensor(device=visual_projection.device),dim=-1).T #(bs*2,312)

        concept_projections = self.concept_projection(torch.mean(visual_patches,dim=1)) # average pooling of visual patch tokens
        concepts = concept_activations + concept_projections #(bs*2,312)
 
        return concepts, self.classifier(concepts)
            #     (bs*2,312)         (bs*2,200) 



"""
###############################################################################################################
                                                    Loss Functions
###############################################################################################################
"""

class ClipLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.logit_scale = 1.0/temperature

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, img1_features, img2_features):
        logits_per_image = self.logit_scale * img1_features @ img2_features.T
        return logits_per_image

    def forward(self, x):
        img1_features, img2_features = x[:,0,:], x[:,1,:]

        img1_features = F.normalize(img1_features, dim=-1)
        img2_features = F.normalize(img2_features, dim=-1)
        
        logits_per_image = self.get_logits(img1_features, img2_features)
        labels = self.get_ground_truth(img1_features.device, logits_per_image.shape[0])
        return F.cross_entropy(logits_per_image, labels)


class ss_concept_loss(nn.Module):
    def __init__(self):
        super(ss_concept_loss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        # softmax normalization of concept score distributions (both predicted and ground-truth) makes them very weak.
        # Concept scores values are typically at the order of 1e-4.
        # a large scale value makes them significant enough to be trained with L1 loss.
        self.scale = 1.0/4e-5
    
    def forward(self, predicted_concepts, class_logits, class_label, concept_label, use_concept_labels):

        classification_loss = self.ce_loss(class_logits, class_label)

        normalized_predicted_concepts = self.scale * F.softmax(predicted_concepts,dim=-1) * use_concept_labels.unsqueeze(-1).expand(-1,312)
        normalized_concept_labels = self.scale * F.softmax(concept_label,dim=-1) * use_concept_labels.unsqueeze(-1).expand(-1,312)
        # Semi-Supervision -->> if use_concept_labels = 0, then both the above terms becomes 0.
        concept_loss = self.l1_loss(normalized_predicted_concepts, normalized_concept_labels)

        return classification_loss, concept_loss


# Trinity loss (combination of 3 loss functions) = contrastive loss + classification loss (supervised) + concept loss (semi-supervised)
class trinity_loss(nn.Module):
    def __init__(self):
        super(trinity_loss, self).__init__()

        self.clip_loss = ClipLoss()
        self.ss_loss = ss_concept_loss()

    def forward(self, predicted_concepts, class_logits, class_label, concept_label, use_concept_labels):
        bs_2,dim = predicted_concepts.shape
        contrasive_loss = self.clip_loss(torch.reshape(predicted_concepts,(int(bs_2/2),2,dim)))
        classification_loss, concept_loss = self.ss_loss(predicted_concepts, class_logits, class_label, concept_label, use_concept_labels)
        return contrasive_loss, classification_loss, concept_loss