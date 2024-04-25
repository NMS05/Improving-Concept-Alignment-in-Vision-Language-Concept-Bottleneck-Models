import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
import numpy as np


"""
###############################################################################################################
                                                        MODELS
###############################################################################################################
"""

# 18 attributes
rival_atributes = [
    'an animal with long-snout', 
    'an animal with  wings', 
    'a vehicle with wheels', 
    'has text written on it', 
    'an animal with  horns', 
    'an animal with floppy-ears', 
    'an animal with ears', 
    'an animal with colored-eyes', 
    'an object or an animal with a tail', 
    'an animal with mane', 
    'an animal with beak', 
    'an animal with hairy coat', 
    'an object with a metallic body', 
    'an object with rectangular shape', 
    'is damp, wet, or watery ', 
    'a long object', 
    'a tall object', 
    'has patterns on it'
    ]


# CLIP VL-CBM
class clip_cbm(nn.Module):
    def __init__(self):
        super(clip_cbm, self).__init__()

        # clip model
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
        for p in self.clip_model.parameters(): p.requires_grad=False
        
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        tokenized_text_concepts = tokenizer(rival_atributes)

        # concept features as tensors. shape = (18,512).
        self.concept_features = self.clip_model.encode_text(tokenized_text_concepts, normalize=True).cpu().numpy()

        # classifer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(18),
            nn.Linear(18,10)
        )
    
    def _get_concept_tensor(self, device):
        return torch.tensor(self.concept_features, requires_grad=False, device=device)
    
    def forward(self, images): 
        visual_features = self.clip_model.encode_image(images, normalize=True)
        dev = visual_features.device
        concept_activations = visual_features @ self._get_concept_tensor(device=dev).T
        return concept_activations, self.classifier(concept_activations) # returns raw concept activation scores, class predictions


# Contrastive Semi-Supervised (CSS) VL-CBM
class css_cbm(nn.Module):
    def __init__(self,):
        super(css_cbm, self).__init__()

        # clip model
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
        self.clip_model.visual.output_tokens = True
        for p in self.clip_model.parameters(): p.requires_grad=False
        
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        tokenized_text_concepts = tokenizer(rival_atributes)

        # concept features as tensors. shape = (18,512). My concepts
        self.concept_features = self.clip_model.encode_text(tokenized_text_concepts).cpu().numpy()

        self.concept_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768,18)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(18),
            nn.Linear(18,10),
        )
    
    def _get_concept_tensor(self, device):
        return torch.tensor(self.concept_features, requires_grad=False, device=device)
    
    def forward(self, image_pairs):

        bs,imgs,channels,h,w = image_pairs.shape
        images = torch.reshape(image_pairs,(bs*imgs,channels,h,w))

        visual_projection, visual_patches = self.clip_model.encode_image(images)
        concept_projections = self.concept_projection(torch.mean(visual_patches,dim=1))

        concept_activations = F.normalize(visual_projection, dim=-1) @ F.normalize(self._get_concept_tensor(device=visual_projection.device),dim=-1).T
        concepts = concept_activations + concept_projections
            #     (bs*2,18)         (bs*2,10)
        return F.sigmoid(concepts), self.classifier(concepts)
    

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
        self.scale = 1.0/1e-4
    
    def forward(self, predicted_concepts, class_logits, class_label, concept_label, use_concept_labels):

        classification_loss = self.ce_loss(class_logits, class_label)

        normalized_predicted_concepts = self.scale * predicted_concepts * use_concept_labels.unsqueeze(-1).expand(-1,18)
        normalized_concept_labels = self.scale * concept_label * use_concept_labels.unsqueeze(-1).expand(-1,18)
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