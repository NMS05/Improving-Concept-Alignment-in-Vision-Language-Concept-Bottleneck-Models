"""
For distributional analysis, run the following code snippets one-by-one

1. extract CLIP concept scores
2. extract CSS concept scores
3. Distributional Analysis of Concept Space
4. t-SNE visualization of 5 random CUB classes
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from torch.utils.data import DataLoader
from data.cub_data import Processed_CUB_Dataset



"""
Load CUB test
"""
test_dataset = Processed_CUB_Dataset(split="test")
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)



"""
1. extract CLIP concept scores
"""
from model.cbms import clip_cbm

# organize concept scores classwise
concept_activations = []
for _ in range(200): concept_activations.append([])

model = clip_cbm()
model.eval()
device = torch.device("cuda")
model.to(device)

for images, labels in test_loader:  

    images = images.to(device)
    con_acv, _ = model(images)

    for ca, lbl in zip(con_acv,labels):
        # print(ca.shape,lbl.item())
        concept_activations[lbl.item()].append(ca.cpu().numpy())
    #     break
    # break

for i,CA in enumerate(concept_activations):
    concept_activations = np.stack(CA, axis=0)
    print(concept_activations.shape)
    np.save('concept_scores/clip_'+str(i)+'.npy', concept_activations)



"""
CSS VL-CBM concept scores
"""
# organize concept scores classwise
concept_activations = []
for _ in range(200): concept_activations.append([])

# same architecture as css vl-cbm, but the forward pass is modified for single images (image pairs required only for for CSS training)
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
    
model = trained_cbm()
model.load_state_dict(torch.load("trained_css_vl_cbm.pth")) # load the css trained weights
device = torch.device("cuda")
model.to(device)

for images, labels in test_loader:  

    images = images.to(device)
    con_acv,_ = model(images)

    for ca, lbl in zip(con_acv,labels):
        ca = ca.detach().cpu().numpy()
        # print(ca.shape,lbl.item())
        concept_activations[lbl.item()].append(ca)
    #     break
    # break

for i,CA in enumerate(concept_activations):
    concept_activations = np.stack(CA, axis=0)
    print(concept_activations.shape)
    np.save('concept_scores/css_'+str(i)+'.npy', concept_activations)





"""
Distributional Analysis of Concept Space
"""

# cacs : class-averaged (ground-truth) concept scores
cacs = torch.tensor(np.load("data/class_averaged_concept_scores.npy"))
print("\nClass Averaged Concept Scores - ",cacs.shape)


########## truthfulness: defined as the L2-distance between predicted and ground-truth concepts   ########## 

classwise_distance = []

for i in range(200):
    gt_center = cacs[i]
    pred_center = np.mean(np.load('concept_scores/css_'+str(i)+'.npy'),axis=0) # for css
    # pred_center = np.mean(np.load('concept_scores/clip_'+str(i)+'.npy'),axis=0) # for clip
    dist = np.linalg.norm(gt_center - pred_center)
    classwise_distance.append(round(dist.item(),3))

print(np.mean(np.asarray(classwise_distance)))


#########  sparseness: measures the intra-class standard deviation of the predicted concepts  ########## 
stds = []
for i in range(200):
    std = np.std(np.load('concept_scores/css_'+str(i)+'.npy'),axis=0) # for css
    # std = np.std(np.load('concept_scores/clip_'+str(i)+'.npy'),axis=0) # for clip
    stds.append(std)
print(np.mean(np.asarray(stds)))


########## discriminability: defined as the L2-distance between inter-class concept score clusters #######

classwise_center = []
for i in range(200): 
    classwise_center.append(np.mean(np.load('concept_scores/css_'+str(i)+'.npy'),axis=0)) # for css
    # classwise_center.append(np.mean(np.load('concept_scores/clip_'+str(i)+'.npy'),axis=0)) # for clip
classwise_center = np.stack(classwise_center,axis=0)
# print(classwise_center.shape)

from scipy.spatial import distance
dist_matrix = distance.cdist(classwise_center,classwise_center)
print(dist_matrix.shape, dist_matrix.mean())





"""
tSNE visualization
"""
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# chose 5 random CUB classes
random_classes = np.random.choice(np.arange(200), size=5, replace=False)

clip_concept_scores = []
css_concept_scores = []
avg_concept_scores = [] # ground-truth scores

unique_class_labels = []
class_labels = [] # class label for every sample

for i, class_num in enumerate(random_classes):

    # load the concept scores of all samples in a class
    clip_scores = np.load(f"concept_scores/clip_{class_num}.npy")
    css_scores = np.load(f"concept_scores/css_{class_num}.npy")
    # print(class_num, clip_scores.shape, css_scores.shape)

    # class averaged concept scores
    avg_concept_scores.append(cacs[class_num])
    unique_class_labels.append(i)

    # load concept scores of 20 random samples per class
    for sample_num in np.random.choice(np.arange(len(clip_scores)), size=20, replace=False):
        # clip
        _clip = clip_scores[sample_num]
        clip_concept_scores.append(_clip)
        # css
        _css = css_scores[sample_num]
        css_concept_scores.append(_css)
        # labels
        class_labels.append(i)

all_concept_scores = clip_concept_scores + css_concept_scores + avg_concept_scores
all_concept_scores = np.asarray(all_concept_scores)

class_labels = np.asarray(class_labels)
unique_class_labels = np.asarray(unique_class_labels)

print(all_concept_scores.shape, class_labels.shape, unique_class_labels.shape)


# Perform t-SNE
tsne = TSNE(n_components=2, max_iter=2000, random_state=0)
embedded = tsne.fit_transform(all_concept_scores)

# Create a colormap
cmap = plt.get_cmap(name='tab10')

# Plot the result
plt.figure(figsize=(8, 8))

# plot clip concept scores (circles)
plt.scatter(
    x=embedded[:len(clip_concept_scores), 0], 
    y=embedded[:len(clip_concept_scores), 1], 
    marker='o', 
    c=class_labels, 
    cmap=cmap, 
    alpha=0.5)

# plot cs concept scores (triangles)
plt.scatter(
    x=embedded[len(clip_concept_scores):-5, 0], 
    y=embedded[len(clip_concept_scores):-5, 1], 
    marker='^', 
    c=class_labels, 
    cmap=cmap, 
    alpha=0.5)

# plot the class averaged ground truth score (bold X)
plt.scatter(
    x=embedded[-5:, 0], 
    y=embedded[-5:, 1], 
    marker='X', 
    c=unique_class_labels, 
    cmap=cmap, 
    alpha=1.0)

plt.title("t-SNE Visualization of Concept Scores")
plt.show()